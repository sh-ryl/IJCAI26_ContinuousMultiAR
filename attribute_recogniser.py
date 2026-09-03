import torch
import torch.nn.functional as F
import numpy as np
from utils import elo, is_default_param
from constants import COLORS, REWARDABLE_ITEMS, RECIPES
from dqn import DQN, DQN_Config
import scipy
from inspect import currentframe, getframeinfo
# from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import multivariate_normal, norm
from scipy.optimize import curve_fit


import os
import re
from copy import deepcopy
import math

from itertools import chain

import pandas as pd

import matplotlib.pyplot as plt

import csv


class AttributeRecogniser(object):

    def __init__(self, attr_list={}, model_temperature=0.5, hypothesis_momentum=0.9999, kl_tolerance=0.0, saved_model_dir=None, dqn_config: DQN_Config = None, show_graph=False, log_dir=None, exp_param=[], max_steps=100):

        # IO Settings
        self.saved_model_dir = saved_model_dir
        self.show_graph = show_graph
        self.first_log_write = True
        self.log_dir = log_dir

        # AR Model Settings
        self.model_temperature = model_temperature
        self.hypothesis_momentum = hypothesis_momentum
        self.kl_tolerance = kl_tolerance
        self.device = torch.device("cuda" if dqn_config.gpu >= 0 else "cpu")

        self.step_number = 0
        self.max_steps = max_steps

        # Load AR observer models
        self.trained_models = []
        self.tm_paths, self.tm_param = self.find_folders(
            self.saved_model_dir, attr_list, exp_param)

        print()
        print("Printing all models found for the observer...")

        # Choose Models to include as observer
        for i in range(len(self.tm_paths)):
            print(
                f"AR model {i}: {self.tm_paths[i]}, Param: {self.tm_param[i]}")

        print()
        print("Do you want to include all models listed above?")
        start = input(
            "Press [Enter] to include all\nPress [C then Enter] to choose a single model\nPress [Other Keys then Enter] to remove multiple models from the list ")
        if start != "":
            if start == "c" or start == "C":
                choice = int(input(
                    "Which model to select? (ONE model number only) "))
                self.tm_paths = [self.tm_paths[choice]]
                self.tm_param = [self.tm_param[choice]]
            else:
                choice = input(
                    "Which models to remove? (Model number separated by comma) ")
                choice = sorted([int(x) for x in choice.split(",")])
                remove_count = 0
                for i in choice:
                    print(
                        f"REMOVING AR model: {self.tm_paths[i-remove_count]}")
                    self.tm_paths.pop(i-remove_count)
                    self.tm_param.pop(i-remove_count)
                    remove_count += 1

            print()
            print("Updated Model List")
            for i in range(len(self.tm_paths)):
                print(
                    f"AR model {i}: {self.tm_paths[i]}, Param: {self.tm_param[i]}")

        # set DQN config for each selected model
        for i in range(len(self.tm_paths)):
            model_file = f'{self.tm_paths[i]}/model.chk'
            checkpoint = torch.load(model_file, map_location=self.device)

            tm_state_size = checkpoint['model_state_dict']['fc1.weight'].size()[
                1]
            tm_dqn_config = dqn_config
            tm_dqn_config.set_state_size(tm_state_size)

            self.trained_models.append(DQN(tm_dqn_config))
            self.trained_models[i].load_state_dict(
                checkpoint['model_state_dict'])
            self.tm_paths[i] = self.tm_paths[i].removeprefix(
                self.saved_model_dir)

        # parameter from the terminal
        self.exp_param = exp_param

        if "uvfa" in self.tm_param[0] and not "multivar" in self.tm_param[0]:
            # exp ver
            if "ability" in self.tm_param[0]:
                self.attr_list = attr_list
                self.uvfa_weight = [x for x in range(
                    100, 1200, int(((1200-100)/20)))]
                self.ar_num = len(self.uvfa_weight)
                print()
                print("Observer Weights:", self.uvfa_weight)
            elif "pref" in self.tm_param[0]:
                # currently only applied for attribute preferences and for 2 weights
                self.uvfa_weight = [(x/10, (10-x)/10)
                                    for x in range(1, 10, 1)]
                self.attr_list = attr_list
                self.ar_num = len(self.uvfa_weight)

            self.dist_param = None

            # old ver
            # self.uvfa_weight = [(x/10, (10-x)/10) for x in range(1, 10, 1)]
            # self.attr_list = attr_list
            # print("UVFA observes these weights", self.uvfa_weight)
            # self.ar_num = len(self.uvfa_weight)
        elif "multivar" in self.tm_param[0]:
            # currently ability and pref only
            ab_num = 20
            ability_arr = [x for x in range(
                100, 1200, int(((1200-100)/ab_num)))]
            pref_arr = [(0.1, 0.1, 0.1, 0.7),
                        (0.1, 0.1, 0.7, 0.1),
                        (0.1, 0.7, 0.1, 0.1),
                        (0.7, 0.1, 0.1, 0.1),
                        (0.2, 0.2, 0.2, 0.4),
                        (0.2, 0.2, 0.4, 0.2),
                        (0.2, 0.4, 0.2, 0.2),
                        (0.4, 0.2, 0.2, 0.2),
                        ]

            self.uvfa_weight = {"ability": ability_arr, "pref": pref_arr}
            self.attr_list = attr_list
            self.ar_num = len(ability_arr) * len(pref_arr)
        else:
            self.ar_num = len(self.trained_models)

        print("AR INIT", self.uvfa_weight, self.attr_list)
        # Init scores for evaluation
        self.tm_dkl_sum = [0] * self.ar_num
        self.tm_dkl_sum_eptotal = [[0 for y in range(
            self.ar_num)]for x in range(max_steps)]  # 100 for max timestep
        self.tm_dkl_step_eptotal = deepcopy(self.tm_dkl_sum_eptotal)

        self.tm_dkl_ravg_prev = [0] * self.ar_num
        self.tm_dkl_ravg_eptotal = deepcopy(self.tm_dkl_sum_eptotal)
        self.tm_dkl_zbc_eptotal = deepcopy(self.tm_dkl_sum_eptotal)

        self.tm_bi_prob = [
            1.0 / int(self.ar_num)] * int(self.ar_num)
        self.tm_bi_prob_eptotal = deepcopy(
            self.tm_dkl_sum_eptotal)
        self.tm_bi_prob_step = [[0 for y in range(
            self.ar_num)]for x in range(max_steps)]

        self.sampling_min_step = [0 for x in range(self.max_steps)]
        self.sampling_max_step = [0 for x in range(self.max_steps)]
        self.alpha_cdf_1_step = [0 for x in range(max_steps)]
        self.alpha_pdf_max_step = [0 for x in range(max_steps)]
        self.alpha_pdf_max_step_pref = [0 for x in range(self.max_steps)]

        self.cdf_step = [[] for x in range(self.max_steps)]
        self.pdf_step = [[] for x in range(self.max_steps)]

    def reset(self):
        self.tm_dkl_sum = [0] * self.ar_num
        self.tm_bi_prob = [
            1.0 / int(self.ar_num)] * int(self.ar_num)
        self.tm_dkl_ravg_prev = [0] * self.ar_num

        self.dist_param = None

        self.sampling_min_step = [0 for x in range(self.max_steps)]
        self.sampling_max_step = [0 for x in range(self.max_steps)]
        self.alpha_cdf_1_step = [0 for x in range(self.max_steps)]
        self.alpha_pdf_max_step = [0 for x in range(self.max_steps)]
        self.alpha_pdf_max_step_pref = [0 for x in range(self.max_steps)]

        self.weight_step = [[] for x in range(self.max_steps)]
        self.cdf_step = [[] for x in range(self.max_steps)]
        self.pdf_step = [[] for x in range(self.max_steps)]

        self.tm_bi_prob_step = [
            [0 for y in range(self.ar_num)]for x in range(self.max_steps)]

    def calculate_action_probs(self, model_no, state):
        state_tensor = torch.from_numpy(state.getRepresentation(ar_obs=True,
                                                                ar_param=self.tm_param[model_no])).float().to(
            self.device).unsqueeze(0)
        q = self.trained_models[model_no].forward(
            state_tensor).cpu().detach().squeeze()
        probs = F.softmax(q.div(self.model_temperature), dim=0)
        return probs

    def perceive(self, state, action: int, outcome, frame_num, print_result, momentum=0.95):
        # things to capture
        # average of the sampled, min max, take for each step
        # average of alpha value which has max pdf
        # average of alpha value when it becomes 1
        # deal with: ValueError, OverflowError, RuntimeError -> ignore episode for now, figure out why later

        bayes_denom = 0
        tm_act_probs = []
        tm_T = []
        tm_dkl_step = []
        ep_step = frame_num % self.max_steps

        if "uvfa" in self.tm_param[0] and "old_ar" not in self.exp_param:
            # CODE FOR DISTRIBUTION FITTING
            model_no = 0

            # STEP 1 get random variables
            # ab lv inference
            sorted_alpha = self.uvfa_weight

            self.weight_step[ep_step] = sorted_alpha

            # STEP 2 calculate posterior prob
            tm_bi_prob_new = [0] * self.ar_num
            eps = 1e-20
            # calculate action probability of each alpha and denominator (marginal prob)
            for i in range(len(self.uvfa_weight["ability"])):
                for j in range(len(self.uvfa_weight["pref"])):
                    mod_id = i * len(self.uvfa_weight["pref"]) + j
                    model_no = 0
                    # currently only 4 weights
                    pref_dic = {self.attr_list[0]: self.uvfa_weight["pref"][j][0],
                                self.attr_list[1]: self.uvfa_weight["pref"][j][1],
                                self.attr_list[2]: self.uvfa_weight["pref"][j][2],
                                self.attr_list[3]: self.uvfa_weight["pref"][j][3]}

                    self.tm_param[0]["multivar"] = {
                        "ability": self.uvfa_weight["ability"][i], "pref": pref_dic}

                    act_probs = self.calculate_action_probs(model_no, state)
                    tm_act_probs.append(act_probs)
                    T = 1

                    # bayes prob
                    if action == 5 and len(state.ab_rating) > 0 and "modbi" in self.exp_param:
                        if "craft_loc" in self.exp_param or "craft_item" in self.exp_param:
                            if outcome[1] in state.ab_rating['craft']:
                                # these ratings can be found in python_agent.py
                                # under first clause of "if "ability" in exp_param:"
                                T = elo(
                                    self.uvfa_weight["ability"][i], state.ab_rating['craft'][outcome[1]])

                        else:
                            T = elo(
                                self.uvfa_weight["ability"][i], state.ab_rating['craft'])

                        if outcome[0] == False:  # if outcome was failure
                            T = 1-T
                    tm_T.append(T)

                    bayes_denom += self.tm_bi_prob[mod_id] * \
                        act_probs[action] * T

            # calculate posterior prob for each alpha values
            for i in range(self.ar_num):
                # bayesian inf
                tm_bi_prob_new[i] = tm_act_probs[i][action] * \
                    self.tm_bi_prob[i] * tm_T[i] / (bayes_denom+eps)

                self.tm_bi_prob_eptotal[frame_num %
                                        self.max_steps][i] += tm_bi_prob_new[i]
                self.tm_bi_prob_step[frame_num %
                                     self.max_steps][i] = tm_bi_prob_new[i]

            if ep_step == self.max_steps-1:
                path = f'{self.log_dir}/bi_prob_ep/'
                if not os.path.exists(path):
                    os.makedirs(path)
                self.generate_output(
                    self.tm_bi_prob_step, path, f'ep_{math.floor(frame_num/self.max_steps)}')

            # Fitting multivariate distribution on ability and prefs
            abilities = np.array(self.uvfa_weight["ability"], float)
            prefs = np.array(self.uvfa_weight["pref"], float)
            y_obs = np.array(tm_bi_prob_new, float)

            try:
                C_hat, mu_hat, Sigma_hat, pdf_norm_fn, info = fit_trunc_logistic_normal_pdf_with_scale(
                    abilities, prefs, y_obs, L=abilities.min(), U=abilities.max()
                )
                self.C_hat = C_hat
                self.mu_hat = mu_hat
                self.Sigma_hat = Sigma_hat
                self.pdf_norm_fn = pdf_norm_fn
                self.info = info
            except:
                C_hat = self.C_hat
                mu_hat = self.mu_hat
                Sigma_hat = self.Sigma_hat
                pdf_norm_fn = self.pdf_norm_fn
                info = self.info

            # get pdf on a large number of random values
            n = 1000
            abilities_s, prefs_s = sample_uniform_ability_and_pref(n, ability_range=(0, 1200),
                                                                   dirichlet_alpha=(
                                                                       1, 1, 1, 1),
                                                                   integer_ability=False, rng=np.random.default_rng(0))

            best_sample = argmax_over_samples(
                abilities_s, prefs_s, pdf_norm_fn, C_hat=C_hat)
            print("Best ability:", best_sample["ability"])
            print("Best preference:", best_sample["pref"])

            self.alpha_pdf_max_step[ep_step] = best_sample["ability"]
            self.alpha_pdf_max_step_pref[ep_step] = best_sample["pref"]

            self.tm_bi_prob = tm_bi_prob_new

        elif "multivar" in self.tm_param[0] and "old_ar" in self.exp_param:
            # calculate denominator
            for i in range(len(self.uvfa_weight["ability"])):
                for j in range(len(self.uvfa_weight["pref"])):
                    mod_id = i * len(self.uvfa_weight["pref"]) + j
                    model_no = 0
                    # currently only 4 weights
                    pref_dic = {self.attr_list[0]: self.uvfa_weight["pref"][j][0],
                                self.attr_list[1]: self.uvfa_weight["pref"][j][1],
                                self.attr_list[2]: self.uvfa_weight["pref"][j][2],
                                self.attr_list[3]: self.uvfa_weight["pref"][j][3]}

                    self.tm_param[0]["multivar"] = {
                        "ability": self.uvfa_weight["ability"][i], "pref": pref_dic}

                    act_probs = self.calculate_action_probs(model_no, state)
                    tm_act_probs.append(act_probs)
                    T = 1

                    # bayes prob
                    if action == 5 and len(state.ab_rating) > 0 and "modbi" in self.exp_param:
                        if "craft_loc" in self.exp_param or "craft_item" in self.exp_param:
                            if outcome[1] in state.ab_rating['craft']:
                                # these ratings can be found in python_agent.py
                                # under first clause of "if "ability" in exp_param:"
                                T = elo(
                                    self.uvfa_weight["ability"][i], state.ab_rating['craft'][outcome[1]])

                        else:
                            T = elo(
                                self.uvfa_weight["ability"][i], state.ab_rating['craft'])

                        if outcome[0] == False:  # if outcome was failure
                            T = 1-T
                    tm_T.append(T)

                    bayes_denom += self.tm_bi_prob[mod_id] * \
                        act_probs[action] * T

            # calculate posterior for each observer type
            tm_bi_prob_new = [0] * self.ar_num
            eps = 1e-20
            if print_result:
                print()
                print("---------------- AR Inference ---------------")
                col = ['No.',
                       'ab lv',
                       'pref',
                       'BI prob',
                       'action prob',
                       'action']
                print(
                    f"{col[0]:3} {col[1]:5} {col[2]:50} {col[3]:10} {col[4]:50} {col[5]} ")
                print()

            for i in range(self.ar_num):
                # bayesian inf
                tm_bi_prob_new[i] = tm_act_probs[i][action] * \
                    self.tm_bi_prob[i] * tm_T[i] / (bayes_denom+eps)

                if print_result:
                    tm_act_probs[i] = [round(float(x), 3)
                                       for x in tm_act_probs[i]]
                    ab_id = math.floor(i/len(self.uvfa_weight["pref"]))
                    ab_lv = self.uvfa_weight["ability"][ab_id]
                    pref_id = i % len(self.uvfa_weight["pref"])
                    pref_dic = {self.attr_list[0][0]: self.uvfa_weight["pref"][pref_id][0],
                                self.attr_list[1][0]: self.uvfa_weight["pref"][pref_id][1],
                                self.attr_list[2][0]: self.uvfa_weight["pref"][pref_id][2],
                                self.attr_list[3][0]: self.uvfa_weight["pref"][pref_id][3]}
                    a = str(i)+'.'
                    b = str(ab_lv)
                    c = str(pref_dic)
                    g = round(float(tm_bi_prob_new[i]), 3)
                    h = tm_act_probs[i]
                    i = int(np.argmax(h))
                    print(
                        f"{a:3} {b:5} {c:50} {g:<10} {str(h):50} {i}")

                # update scores
                # bi prob
                self.tm_bi_prob_eptotal[frame_num %
                                        self.max_steps][i] += tm_bi_prob_new[i]
            self.tm_bi_prob = tm_bi_prob_new

        else:
            # calculate denom
            for i in range(self.ar_num):
                # get action probability from the corresponding model
                model_no = i
                if "uvfa" in self.tm_param[0]:
                    model_no = 0
                    if "ability" in self.tm_param[0]:
                        self.tm_param[0]["uvfa"] = self.uvfa_weight[i]
                    else:  # for preference
                        self.tm_param[0]["uvfa"] = {
                            self.attr_list[0]: self.uvfa_weight[i][0], self.attr_list[1]: self.uvfa_weight[i][1]}

                act_probs = self.calculate_action_probs(model_no, state)
                tm_act_probs.append(act_probs)

                # kl div sum
                dkl_max = 100.0
                dkl_step = min(act_probs[action].pow(-1).log().item(), dkl_max)
                self.tm_dkl_sum[i] += dkl_step
                tm_dkl_step.append(dkl_step)

                bayes_denom += self.tm_bi_prob[i] * act_probs[action]

            # calculate posterior for each observer type
            tm_bi_prob_new = [0] * self.ar_num
            eps = 1e-20
            if print_result:
                print()
                print("---------------- AR Inference ---------------")
                col = ['No.',
                       'trained model paths',
                       'DKL sum',
                       'DKL ravg',
                       'DKL zbc',
                       'BI prob',
                       'action prob',
                       'action']
                print(
                    f"{col[0]:3} {col[1]:30} {col[2]:10} {col[3]:10} {col[4]:10} {col[5]:10} {col[6]:50} {col[7]} ")
                print()

            for i in range(self.ar_num):
                # kl div running avg
                dkl_ravg = momentum * \
                    self.tm_dkl_ravg_prev[i] + (1 - momentum) * tm_dkl_step[i]
                self.tm_dkl_ravg_prev[i] = dkl_ravg

                # kl div zero bias corrected
                denom = 1 - pow(momentum, ((frame_num % self.max_steps) + 1))
                dkl_zbc = dkl_ravg/denom

                # bayesian inf
                tm_bi_prob_new[i] = tm_act_probs[i][action] * \
                    self.tm_bi_prob[i] / bayes_denom+eps

                if print_result:
                    tm_act_probs[i] = [round(float(x), 3)
                                       for x in tm_act_probs[i]]
                    a = str(i)+'.'
                    b = str(
                        self.uvfa_weight[i]) if "uvfa" in self.tm_param[0] else self.tm_paths[i]
                    c = round(self.tm_dkl_sum[i], 3)
                    d = round(dkl_ravg, 3)
                    e = round(dkl_zbc, 3)
                    f = round(float(tm_bi_prob_new[i]), 3)
                    g = tm_act_probs[i]
                    h = int(np.argmax(g))
                    print(
                        f"{a:3} {b:30} {c:<10} {d:<10} {e:<10} {f:<10} {str(g):50} {h}")

                # update scores
                # dkl sum
                self.tm_dkl_sum_eptotal[frame_num %
                                        self.max_steps][i] += self.tm_dkl_sum[i]
                self.tm_dkl_step_eptotal[frame_num %
                                         self.max_steps][i] += tm_dkl_step[i]

                # dkl ravg
                self.tm_dkl_ravg_eptotal[frame_num %
                                         self.max_steps][i] += dkl_ravg

                # dkl zbc
                self.tm_dkl_zbc_eptotal[frame_num %
                                        self.max_steps][i] += dkl_zbc
                # bi prob
                self.tm_bi_prob_eptotal[frame_num %
                                        self.max_steps][i] += tm_bi_prob_new[i]
            self.tm_bi_prob = tm_bi_prob_new

    def find_folders(self, directory, required_objects, exp_param):
        c_tm_path = []
        c_tm_attr_dic = []
        c_tm_param = []
        param_set = {"uvfa", "multivar", "limit",
                     "belief", "ability", "pref",
                     "pref_ability"}

        # Iterate through the directory
        for root, dirs, files in os.walk(directory):
            # Get the directory name before the subdirectory
            # to split "\" because windows don't follow unix style
            root_split = list(chain.from_iterable(
                [x.split("\\") for x in root.split('/')]))

            for folder_name in dirs:
                obj_list = []
                weight_list = []
                # filter out other folders based on object list
                if "uvfa" in exp_param or "multivar" in exp_param:
                    parts = folder_name.split('_')
                    obj_list = [
                        part for part in parts if part in REWARDABLE_ITEMS]
                else:
                    # currently regex works if number is less than 1 with comma or more than 1 without comma
                    pattern = r'([a-zA-Z]+_-*\d+\.?\d*)'
                    result = re.findall(pattern, folder_name)
                    if result:
                        matches = [x.split('_') for x in result]
                        obj_list = [x[0] for x in matches]
                        weight_list = [float(x[1]) for x in matches]

                if "level" in obj_list:
                    obj_list.remove("level")

                if "discount" in obj_list:
                    i = obj_list.index("discount")
                    obj_list.pop(i)
                    weight_list.pop(i)

                # THIS ASSUMES THAT LABELS WILL ALWAYS BE POSITIONED AS THE LAST PARAM
                # if last param is a label, and label is in folder name, keep the folder
                if is_default_param(exp_param[-1]) == False and exp_param[-1] in folder_name:
                    1
                # if last param is not a label, keep the folder
                elif is_default_param(exp_param[-1]) == True:
                    1
                else:
                    continue

                if set(obj_list) == set(required_objects):
                    tm_param = {}
                    tm_attr_dic = {}

                    for param in param_set.intersection(root_split):
                        tm_param[param] = ''

                    if "uvfa" not in exp_param and "multivar" not in exp_param:
                        # get attribute dic
                        for obj in required_objects:
                            tm_attr_dic[obj] = weight_list[obj_list.index(
                                obj)]

                    if "pref" in tm_param:
                        tm_param['pref'] = []

                    if "belief" in tm_param:
                        pattern = r'_hidden_([a-zA-Z]*)_([a-zA-Z]*)'
                        result = re.findall(pattern, folder_name)[0]
                        tm_param['belief'] = result

                    if "ability" in tm_param:
                        pattern = r'_level_(\d+)_uniform'
                        result = re.findall(pattern, folder_name)
                        tm_param['ability'] = result

                    if "pref_ability" in tm_param:
                        tm_param["multivar"] = ''
                        tm_param['ability'] = ''
                        tm_param['pref'] = ''
                        tm_param.pop("pref_ability")

                    if folder_name.find("elo_lens") != -1:
                        tm_param["elo_lens"] = ''

                    if not bool(re.search(r'\d', folder_name)):
                        tm_param["uvfa"] = ''

                    c_tm_path.append(root+'/'+folder_name)
                    c_tm_param.append(tm_param)
                    c_tm_attr_dic.append(tm_attr_dic)

        if "uvfa" not in exp_param and "multivar" not in exp_param:
            combined = zip(c_tm_attr_dic,
                           c_tm_path, c_tm_param)
            sorted_combined = sorted(
                combined, key=lambda x: x[0][required_objects[0]])

            # Extract sorted combined tuples
            c_tm_attr_dic, c_tm_path, c_tm_param = zip(
                *sorted_combined)

        return list(c_tm_path), list(c_tm_param)

    def get_inference(self):
        temp_min = max(self.tm_dkl_sum)
        temp_min_id = 0

        for i in range(self.ar_num):
            if self.tm_dkl_sum[i] < temp_min:
                temp_min = self.tm_dkl_sum[i]
                temp_min_id = i

        return temp_min_id

    def get_result(self, max_frame, attr_str, ag_param={}, item_count_eptotal={}):
        avg_bi_prob = []

        ep_num = max_frame/self.max_steps
        for step in range(self.max_steps):
            savg_bi_prob = [
                ep_total/ep_num for ep_total in self.tm_bi_prob_eptotal[step]]

            avg_bi_prob.append(savg_bi_prob)

        param_str = ''
        for param in sorted(ag_param.keys()):
            param_str += '_' + param
            if str(ag_param[param]) != '':
                param_str += '_' + str(ag_param[param])

        path = self.log_dir + "/" + attr_str + param_str + "_"

        ar_result = [avg_bi_prob]
        ar_result_str = ["avg_bi_prob"]

        if "uvfa" in self.tm_param[0]:
            legend = self.uvfa_weight
        else:
            legend = self.tm_paths
        for i in range(len(ar_result)):
            self.generate_output(
                ar_result[i], path, ar_result_str[i])

    def generate_output(self, data, path, fname):
        data = np.asarray(data)
        np.savetxt(path+fname+".csv", data, delimiter=",")

        df = pd.DataFrame(data=data)

        if "multivar" not in self.tm_param[0]:
            df.columns = self.uvfa_weight
        else:
            col_name = []
            for i in self.uvfa_weight["ability"]:
                for j in self.uvfa_weight["pref"]:
                    col_name.append((i, j))

            df.columns = col_name

            # Calculate mean probability for each column
            sorted_columns = df.mean().sort_values(ascending=False)
            sorted_column_ids = sorted_columns.index.tolist()
            print(sorted_column_ids)

            # Get the top 20 columns
            # get first 20 column names
            top_20_cols = sorted_columns.index[:20]
            df = df[top_20_cols]  # keep only these columns

        # plt.plot(df.index, df, label=df.columns)
        # plt.legend()

        fig, ax = plt.subplots()
        for i, column in enumerate(df.columns):
            ax.plot(df.index, df[column], label=column, color=COLORS[i])

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig_path = path + fname + ".jpg"
        plt.savefig(fig_path)

        plt.clf()


def pdf_on_pairs(ability_samp, pref_samp, pdf_norm_fn, C_hat=None, batch=5000):
    """
    ability_samp : (n,) array
    pref_samp    : (n,4) array, rows sum to 1
    Returns:
      pdf_norm : (n,) normalized pdf values f(a_i, p_i)
      pdf_scaled (optional): (n,) if C_hat is given, C_hat * f(a_i, p_i)
    """
    ability_samp = np.asarray(ability_samp, float).ravel()
    pref_samp = np.asarray(pref_samp, float)
    assert ability_samp.shape[0] == pref_samp.shape[0]
    n = ability_samp.shape[0]

    vals = np.empty(n, float)
    s = 0
    while s < n:
        e = min(s + batch, n)
        A = ability_samp[s:e]
        P = pref_samp[s:e]
        # pdf_norm_fn returns an (len(A) × len(P)) grid; take diagonal for pairwise
        M = pdf_norm_fn(A, P)
        vals[s:e] = np.diag(M)
        s = e

    if C_hat is None:
        return vals
    else:
        return vals, C_hat * vals


def argmax_over_samples(ability_samp, pref_samp, pdf_norm_fn, C_hat=None):
    out = pdf_on_pairs(ability_samp, pref_samp, pdf_norm_fn, C_hat=C_hat)
    if C_hat is None:
        pdf_vals = out
    else:
        pdf_norm, pdf_scaled = out
        pdf_vals = pdf_norm  # argmax is same with/without scaling by constant C_hat

    k = int(np.argmax(pdf_vals))
    return {
        "index": k,
        "ability": float(ability_samp[k]),
        "pref": pref_samp[k].astype(float),
        "pdf_norm": float(pdf_vals[k]),
        **({"pdf_scaled": float(C_hat * pdf_vals[k])} if C_hat is not None else {})
    }


def sample_uniform_ability_and_pref(n, ability_range=(0.0, 1200.0),
                                    dirichlet_alpha=(1, 1, 1, 1),
                                    integer_ability=False, rng=None):
    """
    Draw n samples:
      ability ~ Uniform[ability_range]
      pref    ~ Dirichlet(alpha) (lies on simplex, sum=1, each >0)

    integer_ability=True -> return integer abilities in the given range.
    """
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = map(float, ability_range)

    if integer_ability:
        # integers in [ceil(lo), floor(hi)] inclusive
        lo_i = int(np.ceil(lo))
        hi_i = int(np.floor(hi))
        abilities = rng.integers(lo_i, hi_i + 1, size=n).astype(float)
    else:
        abilities = rng.uniform(lo, hi, size=n)

    prefs = rng.dirichlet(alpha=np.asarray(
        dirichlet_alpha, float), size=n)  # shape (n,4)
    return abilities, prefs

# --- ALR transform (4 -> 3) ---


def alr(p):
    p = np.asarray(p, float)
    if p.shape[-1] != 4:
        raise ValueError("p must have length 4.")
    if np.any(p <= 0):
        raise ValueError("All p_i must be > 0 for ALR.")
    ref = p[..., -1]
    return np.log(p[..., :3] / ref[..., None])  # -> (...,3)

# --- Cholesky parameterization for Sigma (SPD) ---


def _unpack_theta_noC(theta, d=4):
    """
    theta = [mu(d), diag_logs(d), lower_offdiag(d*(d-1)/2)]  ->  mu, Sigma
    Sigma = L L^T with L lower-tri, diag = exp(diag_logs).
    """
    theta = np.asarray(theta, float)
    idx = 0
    mu = theta[idx:idx+d]
    idx += d
    diag_logs = theta[idx:idx+d]
    idx += d
    L = np.zeros((d, d))
    np.fill_diagonal(L, np.exp(diag_logs))
    for i in range(1, d):
        for j in range(i):
            L[i, j] = theta[idx]
            idx += 1
    Sigma = L @ L.T
    return mu, Sigma


def _chol_init_from_grid(abilities, prefs4):
    Zs = []
    for a in np.asarray(abilities, float):
        for p in np.asarray(prefs4, float):
            Zs.append(np.r_[np.log(a), alr(p)])
    Z = np.vstack(Zs)  # (A*P, 4)
    mu0 = Z.mean(axis=0)
    S = np.cov(Z.T) + 1e-6*np.eye(4)
    L0 = np.linalg.cholesky(S)
    diag_logs0 = np.log(np.diag(L0))
    offs0 = [L0[i, j] for i in range(1, 4) for j in range(i)]
    return np.concatenate([mu0, diag_logs0, np.array(offs0)])

# --- Normalized joint PDF f_theta(x,p) (not scaled by C) ---


def _pdf_norm(ability, p1, p2, p3, p4, theta_noC, Lx, Ux):
    x = np.asarray(ability, float)
    P = np.stack([p1, p2, p3, p4], axis=-1).astype(float)
    # softly renormalize tiny drift
    s = P.sum(axis=1, keepdims=True)
    P = P / s
    if np.any(P <= 0):  # ALR requires strictly positive components
        raise ValueError("All p_i must be > 0.")

    y = alr(P)                         # (n,3)
    z = np.column_stack([np.log(x), y])  # (n,4)

    mu, Sigma = _unpack_theta_noC(theta_noC, d=4)
    mvn = multivariate_normal(mean=mu, cov=Sigma)

    # Jacobian for (x,p)->(log x, alr(p)): (1/x) * (1/prod p_i)
    jac = (1.0 / x) * (1.0 / np.prod(P, axis=1))
    base = mvn.pdf(z) * jac

    # ability truncation normalization only
    mu1, s11 = mu[0], Sigma[0, 0]
    sd1 = np.sqrt(max(s11, 1e-12))
    a = (np.log(Lx) - mu1) / sd1
    b = (np.log(Ux) - mu1) / sd1
    denom = np.clip(norm.cdf(b) - norm.cdf(a), 1e-300, None)

    out = base / denom
    out[(x < Lx) | (x > Ux)] = 0.0
    return out

# --- Fit C and theta with curve_fit to y_i = C * f_theta(x_i, p_i) ---


def fit_trunc_logistic_normal_pdf_with_scale(abilities, prefs4, y_obs, L=None, U=None, p0_theta=None, C0=None, maxfev=10000):
    """
    abilities : (A,)              positive ability grid
    prefs4    : (P,4)             each row positive and summing to 1
    y_obs     : (A*P,)            observed values proportional to the true pdf
    L, U      : floats             ability truncation; if None, inferred from data
    p0_theta  : ndarray            initial theta (without C); if None, heuristic
    C0        : float              initial scale; if None, estimated from p0
    Returns:
      C_hat, mu_hat(4,), Sigma_hat(4x4), pdf_norm_fn((A,), (P,4))->(A,P), info
    """

    # validate input
    abilities = np.asarray(abilities, float)
    P = np.asarray(prefs4, float)
    A, Pm = abilities.size, P.shape[0]
    y = np.asarray(y_obs, float).ravel()
    if y.size != A * Pm:
        raise ValueError("y_obs must have length len(abilities)*len(prefs4).")

    if L is None:
        L = float(abilities.min())
    if U is None:
        U = float(abilities.max())
    if not (np.isfinite(L) and np.isfinite(U) and L > 0 and U > L):
        raise ValueError("Require 0 < L < U for ability truncation.")

    # build flattened data arrays: Assemble xdata in nested-loop order
    a_vec = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    for a in abilities:
        for q in P:
            a_vec.append(a)
            p1.append(q[0])
            p2.append(q[1])
            p3.append(q[2])
            p4.append(q[3])
    a_vec = np.array(a_vec, float)
    p1 = np.array(p1, float)
    p2 = np.array(p2, float)
    p3 = np.array(p3, float)
    p4 = np.array(p4, float)

    # Initial params
    if p0_theta is None:
        # initial guess for mu and Sigma
        p0_theta = _chol_init_from_grid(abilities, P)
    if C0 is None:
        # initial scale, estimated by projecting y onto the model at p0
        # quick LS estimate of C given p0: C = (y⋅m)/(m⋅m)
        m0 = _pdf_norm(a_vec, p1, p2, p3, p4, p0_theta, L, U)
        C0 = float(np.maximum(1e-12, np.dot(y, m0) / np.dot(m0, m0)))

    # Model for curve_fit: params = [C, theta_noC...]
    def _model(xdata, C, *theta_noC):
        a, q1, q2, q3, q4 = xdata
        return C * _pdf_norm(a, q1, q2, q3, q4, theta_noC, L, U)

    xdata = (a_vec, p1, p2, p3, p4)

    # Build bounds: C>0; others unconstrained (SPD via Cholesky)
    n_theta = p0_theta.size
    lower = np.r_[1e-12, np.full(n_theta, -np.inf)]
    upper = np.r_[np.inf,  np.full(n_theta,  np.inf)]

    p0_full = np.r_[C0, p0_theta]
    popt, pcov = curve_fit(_model, xdata, y, p0=p0_full,
                           bounds=(lower, upper), maxfev=maxfev)

    C_hat = float(popt[0])
    mu_hat, Sigma_hat = _unpack_theta_noC(popt[1:], d=4)

    # Normalized PDF callable (no scale)
    def pdf_norm_fn(abilities_in, prefs_rows):
        A_in = np.asarray(abilities_in, float).ravel()
        P_in = np.asarray(prefs_rows, float)
        AA, JJ = np.meshgrid(A_in, np.arange(P_in.shape[0]), indexing='ij')
        a_flat = AA.ravel()
        p_mat = P_in[JJ.ravel()]
        vals = _pdf_norm(
            a_flat, p_mat[:, 0], p_mat[:, 1], p_mat[:, 2], p_mat[:, 3], popt[1:], L, U)
        return vals.reshape(A_in.size, P_in.shape[0])

    info = {"pcov": pcov,
            "theta_hat": popt[1:], "C_hat": C_hat, "L": L, "U": U}
    return C_hat, mu_hat, Sigma_hat, pdf_norm_fn, info
