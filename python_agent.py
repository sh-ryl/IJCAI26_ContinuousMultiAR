import os
import random
import sys
import math
from time import gmtime, strftime

import numpy as np
import decimal
import scenario
import pandas as pd

from environment import CraftWorld
from constants import Action, REWARDABLE_ITEMS, COLORS, RECIPES
from utils import is_default_param

from neural_q_learner import NeuralQLearner
from dqn import DQN_Config

from attribute_recogniser import AttributeRecogniser

import torch
import torch.nn.functional as F  # used for generating action probabilities

from copy import deepcopy

import matplotlib.pyplot as plt

ctx = decimal.Context()
ctx.prec = 20

# region HELPER FUNC


def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')


def attr_dic_to_str(attr_dic, inc_weight):
    # *S* why is this function outside of the class?
    attr_dic_keys = list(attr_dic.keys())
    attr_str = attr_dic_keys[0]
    if inc_weight:
        attr_str += "_" + str(attr_dic[attr_dic_keys[0]])
        for i in range(1, len(attr_dic_keys)):
            attr_str += "_" + attr_dic_keys[i] + \
                "_" + str(attr_dic[attr_dic_keys[i]])
    else:
        for i in range(1, len(attr_dic_keys)):
            attr_str += "_" + attr_dic_keys[i]

    return attr_str


def reset_all():

    global agent, total_reward, num_trials, seed, state, ar_obs, AR

    num_trials += 1

    # Only reset the environment seed when we're up to a new agent combination (to remove bias from the evaluation).
    if num_seeds != -1:
        seed = (seed + 1) % num_seeds
    else:
        seed = random.randrange(sys.maxsize)

    if agent_params["test_mode"]:
        print("\nSetting environment seed = " + str(seed) + "...")

    total_reward = 0

    # reset ability level
    if "ability" in exp_param and uvfa:
        # choosing a value from a fixed list
        # p_ratings = [100, 500]
        # r_id = random.randrange(0, 100) % len(p_ratings)
        # ab_rating['player'] = p_ratings[r_id]

        # choosing a value from a range
        if not agent_params["test_mode"]:
            ab_rating['player'] = np.random.uniform(0, 1200)

    agent.reset(
        attr_dic, current_scenario["externally_visible_attr_sets"][0], model_file)
    # will only reset with model file if it's on evaluation mode

    env.ab_rating = ab_rating
    env.state.ab_rating = ab_rating
    state = env.reset([agent], seed)

    if ar_obs:
        AR.reset()

    # Give starting items (if applicable).
    for i in range(len([agent])):
        for item, count in current_scenario["starting_items"][i].items():
            state.inventory[i][item] = count

    # Make the tasks easier at the beginning of training by gifting some starting items (gradually phased out).
    if not agent_params["test_mode"]:
        start_with_item_pr = 0.5 * max(1.0 - frame_num / 1000000, 0)
        for k, v in state.inventory[0].items():
            if np.random.uniform() < start_with_item_pr:
                state.inventory[0][k] += 1


def readjust_ep(e):
    print("ERROR type:", type(e))
    print("current frame num", frame_num)
    ep_num = math.floor(frame_num/max_steps)
    frame_num = ep_num * max_steps
    print("adjusted frame num", frame_num)
    print("restarting episode")
    reset_all()


def print_out(text):
    if co:
        with open(co_file, 'a') as fd:
            fd.write(text+'\n')
    else:
        print(text)

# endregion


# region EXP PARAM
if len(sys.argv) < 2:
    print('Usage:', sys.argv[0], 'scenario')
    sys.exit()

if sys.argv[1] not in scenario.scenarios:
    print("Unknown scenario: " + sys.argv[1])
    sys.exit()

env_name = "cooperative_craft_world"

size = (7, 7)

current_scenario = scenario.scenarios[sys.argv[1]]

attr_sets = current_scenario["attr_sets"]
attr_dic = attr_sets[0]

if "externally_visible_attr_sets" not in current_scenario:
    current_scenario["externally_visible_attr_sets"] = attr_sets

if "num_spawned" not in current_scenario:
    current_scenario["num_spawned"] = scenario.scenarios["default"]["num_spawned"]

if "regeneration" not in current_scenario:
    current_scenario["regeneration"] = scenario.scenarios["default"]["regeneration"]

if "starting_items" not in current_scenario:
    current_scenario["starting_items"] = scenario.scenarios["default"]["starting_items"]

if "hidden_items" not in current_scenario:
    current_scenario["hidden_items"] = scenario.scenarios["default"]["hidden_items"]

agent_params = {}

# TODO: setup scrum output to check if everything is set correctly -- should set up a flag to print out just for scrum!
real_path = os.path.dirname(os.path.realpath(__file__))

co = False
if "co" in sys.argv:
    sys.argv.remove('co')
    co = True
    co_path = real_path + "/check_out"

    if not os.path.exists(co_path):
        os.makedirs(co_path)

    co_file = f'{co_path}/{strftime("%Y%m%d-%H%M%S", gmtime())}.txt'

max_steps = 100
num_episode = 100

if sys.argv[1] == "train":
    agent_params["test_mode"] = False
    n_agents = 1
    num_seeds = -1  # Use random seed
    if torch.cuda.is_available():
        gpu = 0
        print_out("Training using CUDA")
    else:
        gpu = -1
        print_out("Training using CPU")

else:
    max_steps = 50
    agent_params["test_mode"] = True
    n_agents = 1  # 2 # Change for this code since we are only doing single agent AR
    num_seeds = num_episode  # Use random seed
    gpu = -1
    print_out("Testing using CPU")

# exp_param flags
exp_param = sys.argv[2::]
exp_param_path = ''
ab_rating = {}

env_render = False
ar_obs = False
print_result = False
belief = False
limit = False
ability = False
uvfa = False

# to label a certain training model
# assumes that the last param is not in default param, it becomes the model's label
custom_param = ""
for x in exp_param:
    if not is_default_param(x):
        custom_param += x

if "AR" in exp_param:
    ar_obs = True
    ar_out_param = {}
    log_label = input("Add AR log label? (Press ENTER to leave it blank.): ")
    if log_label != "":
        log_label = "_" + log_label
    exp_param.remove('AR')
    print_out("AR Observer is ON")
else:
    print_out("AR Observer is OFF")

if "render" in exp_param:
    env_render = True
    exp_param.remove('render')
    print_out("Rendering environment ON")
    if ar_obs:
        print_result = True
        print_out("Printing result from AR")
else:
    print_out("Rendering environment OFF")

if "limit" in exp_param:
    limit = True
    print_out(
        "Max inventory for collectible items (grass, iron, and wood) is LIMITED to 1")
    exp_param_path += 'limit/'
    if ar_obs:
        ar_out_param['limit'] = ''
else:
    print_out("Max inventory is 999 for all ingredients")

if "uvfa" in exp_param:
    uvfa = True
    print_out("UVFA is ON")
    if ar_obs:
        ar_out_param['uvfa'] = ''
else:
    print_out("UVFA is OFF")

if "pref" in exp_param:
    pref = True
    exp_param_path += 'pref'
    if "multivar" in exp_param:
        exp_param_path += '_'
    else:
        exp_param_path += '/'
    if ar_obs:
        ar_out_param['pref'] = ''

if "belief" in exp_param:
    belief = True
    print_out("Using hidden items")
    exp_param_path += 'belief'
    if "multivar" in exp_param:
        exp_param_path += '_'
    else:
        exp_param_path += '/'
    if ar_obs:
        ar_out_param['belief'] = ''

if "ability" in exp_param:  # the order of pref > belief > ability assumes that ability is always the last one
    ability = True
    print_out("Using ability")
    exp_param_path += 'ability/'
    if not uvfa or agent_params["test_mode"]:
        ab_rating['player'] = int(
            input("Enter ability level for player (Ground Truth, values between 0-1200): "))
    elif uvfa:
        ab_rating['player'] = np.random.uniform(0, 1200)
        print_out(
            "Ability rating for player is set to random for each episode (UVFA flag on)")
    if "craft_loc" in exp_param:  # different crafting difficulty based on location
        ab_rating['craft'] = {'workbench': 100,
                              'factory': 100,
                              'toolshed': 500}
    elif "craft_item" in exp_param:
        ab_rating['craft'] = {"plank": 100,  # wood
                              "bridge": 400,  # iron, wood
                              "axe": 700,  # iron, stick
                              }
        max_steps = 200
    else:
        ab_rating['craft'] = 100
    ab_rating['max'] = 1200
    ab_rating['min'] = 0
    print_out(
        f"Ability rating for craft action is set to {ab_rating['craft']}")
    if ar_obs:
        # NOTE FOR FUTURE SELF: might need to set for uvfa observer differently
        # currently this value comes from uvfa bracket which is randomized
        # if we want a specific value, code needs to be readjusted

        ar_out_param['ability'] = ab_rating['player']

if "degrade" in exp_param:
    print_out("Degrading tools is ON")
    ab_rating['collect'] = 100
    if ar_obs:
        ar_out_param['degrade'] = ''

if "incentive" in exp_param:
    print_out("Incentive is ON")
    if ar_obs:
        ar_out_param['incentive'] = ''

# endregion

# region ENV INIT
env = CraftWorld(current_scenario, size=size, n_agents=n_agents, allow_no_op=False, render=env_render,
                 ingredient_regen=current_scenario["regeneration"], max_steps=max_steps, exp_param=exp_param, ab_rating=ab_rating, test_mode=agent_params["test_mode"])
# endregion

agent_params["agent_type"] = "dqn"

# OPTIMIZER settings
agent_params["adam_lr"] = 0.000125
agent_params["adam_eps"] = 0.00015
agent_params["adam_beta1"] = 0.9
agent_params["adam_beta2"] = 0.999

# region I/O SETTINGS
# FILE I/O settings

# Agent I/O settings
ag_models_root = '/mod/ag/'
ag_models_folder = ag_models_root + exp_param_path

# saving/loading agent model for specific reward weightings
result_folder = ag_models_folder + attr_dic_to_str(attr_dic, inc_weight=True)
# and not ar_obs:  # COMMENT OUT "and not ar_obs" TO USE UVFA MODEL FOR AGENT
if uvfa:
    result_folder = ag_models_folder + \
        attr_dic_to_str(attr_dic, inc_weight=False)
if belief:
    result_folder += "_hidden"
    for hi in current_scenario["hidden_items"][0]:
        result_folder += "_" + hi
if ability:
    if uvfa:
        result_folder += f"_level_random_uniform"
        # random rating for player
        # uniform rating for all crafting skills
    else:
        result_folder += f"_level_{ab_rating['player']}_uniform"
        # uniform rating for all crafting skills
        # will look for ways to store skills with different difficulty ratings
if "incentive" in exp_param:
    result_folder += f"_incentive"
if "degrade" in exp_param:
    result_folder += f"_degrade"
if "elo_lens" in exp_param:
    result_folder += f"_elo_lens"
if custom_param != '':
    result_folder += "_" + custom_param

agent_params["log_dir"] = real_path + f'{result_folder}/'
if not os.path.exists(agent_params["log_dir"]) and not ar_obs:
    os.makedirs(agent_params["log_dir"])
    print_out(f"Created new folder: {agent_params['log_dir']}")
print_out(f"Agent model loaded: {result_folder}")

if not agent_params["test_mode"]:
    training_scores_file = "training_scores.csv"
    with open(agent_params["log_dir"] + training_scores_file, 'a') as fd:
        fd.write('Frame,Score\n')

model_file = real_path + result_folder + \
    "/model.chk"  # To be used in eval mode only

if agent_params["test_mode"]:
    testing_scores_file = 'testing_scores.csv'
    with open(agent_params["log_dir"] + testing_scores_file, 'w') as fd:
        fd.write('seed,agent,agent_score\n')

agent_params["saved_model_dir"] = os.path.dirname(
    os.path.realpath(__file__)) + '/saved_models/'

# Observer (AR) I/O Settings
if ar_obs:
    ar_models_root = '/mod/ag/'
    ar_models_folder = ar_models_root
    model_dir = real_path + ar_models_folder
    print_out(f"AR model folder: {ar_models_folder}")

    if uvfa:
        inc_weight = False
    else:
        inc_weight = True

    attr_log_path = real_path + '/mod/ar_log/' + \
        attr_dic_to_str(
            attr_dic, inc_weight=inc_weight) + "_" + custom_param + log_label
    if not os.path.exists(attr_log_path):
        os.makedirs(attr_log_path)
# endregion

# region DQN Settings
agent_params["dqn_config"] = DQN_Config(
    env.observation_space.shape[0], env.action_space.n, gpu=gpu, noisy_nets=False, n_latent=64)

agent_params["n_step_n"] = 1
agent_params["max_reward"] = 2.0  # 1.0 # Use float("inf") for no clipping
agent_params["min_reward"] = -2.0  # -1.0 # Use float("-inf") for no clipping
agent_params["exploration_style"] = "e_greedy"  # e_greedy, e_softmax
if agent_params["test_mode"]:
    agent_params["exploration_style"] = "e_softmax"  # e_greedy, e_softmax
agent_params["softmax_temperature"] = 0.05

agent_params["ep_start"] = 1
agent_params["ep_end"] = 0.01
agent_params["ep_endt"] = 1000000
agent_params["discount"] = 0.95

if "craft_item" in exp_param:
    agent_params["ep_end"] = 0.05
    agent_params["ep_endt"] = 2000000
    agent_params["discount"] = 0.97

# To help with learning from very sparse rewards initially
agent_params["mixed_monte_carlo_proportion_start"] = 0.2
agent_params["mixed_monte_carlo_proportion_endt"] = 1000000

if agent_params["test_mode"]:
    agent_params["learn_start"] = -1  # Indicates no training
else:
    agent_params["learn_start"] = 50000

agent_params["update_freq"] = 4
agent_params["n_replay"] = 1
agent_params["minibatch_size"] = 32
agent_params["target_refresh_steps"] = 10000
agent_params["show_graphs"] = False
agent_params["graph_save_freq"] = 25000

# For training methods that require n step returns, set the below to True.
agent_params["post_episode_return_calcs_needed"] = True

# can be put together in evaluation setting since transition params is not using it
agent_params["eval_ep"] = 0.01

transition_params = {}
transition_params["agent_params"] = agent_params
transition_params["replay_size"] = 1000000
transition_params["bufferSize"] = 512
# endregion

# region EVAL SETTINGS
eval_freq = 250000  # As per Rainbow paper
eval_steps = 125000  # As per Rainbow paper
# Don't start evaluating until gifted items at the start of training are phased out.
eval_start_time = 1000000

eval_running = False  # different than test_mode in agent config
frame_num = 0

# region FRAMES
max_training_frames = 10000000  # 999999999
if uvfa and not agent_params["test_mode"]:  # if uvfa:
    max_training_frames = 999999999
elif agent_params["test_mode"]:  # if ar_obs:
    max_training_frames = num_episode * max_steps  # 10000
steps_since_eval_ran = 0
steps_since_eval_began = 0
eval_total_score = 0
eval_total_episodes = 0
best_eval_average = float("-inf")
episode_done = False

print_out(f"MAX steps {max_steps}")
print_out(f"MAX training frame {max_training_frames}")
print_out(f"Total Episode {max_training_frames/max_steps}")
# endregion
# endregion

# region AGENT INIT
# Initialise agent
agent = NeuralQLearner("Q_learner", agent_params, transition_params)
agent_combos = [[agent]]

reward = 0
total_reward = 0

num_trials = 0

# So that we reset seeds during the first call of reset_all().
seed = -1
state = None
# endregion

# region AR INIT

if ar_obs:
    # creates a separate config for ar and agent
    ar_dqn_config = deepcopy(agent_params["dqn_config"])
    AR = AttributeRecogniser(attr_list=list(attr_dic.keys()), saved_model_dir=model_dir,
                             dqn_config=ar_dqn_config, log_dir=attr_log_path, exp_param=exp_param, max_steps=max_steps)
    sampling_min_sum = [0 for x in range(max_steps)]
    sampling_max_sum = [0 for x in range(max_steps)]
    alpha_cdf_1_sum = [0 for x in range(max_steps)]
    alpha_pdf_max_sum = [0 for x in range(max_steps)]
    pdf_max_ep = [[] for x in range(max_steps)]
    pdf_max_mape_ep = [[] for x in range(max_steps)]
    pdf_max_ep_pref = [[[] for x in range(max_steps)]
                       for j in range(len(attr_dic.keys()))]
    cdf_1_ep = [[] for x in range(max_steps)]
    max_mape = 0
# endregion

reset_all()

# region MAIN LOOP
# input("Start?")
# print("---------------------------------------------")

item_count_eptotal = {k: [0] * max_steps for k in REWARDABLE_ITEMS}
action_count_eptotal = {k.value: [0] * max_steps for k in Action}

while frame_num < max_training_frames:
    if frame_num % max_steps == 0 and agent_params["test_mode"]:
        print(state.objects)
    prev_state = deepcopy(state)
    a = agent.perceive(reward, prev_state, episode_done, eval_running)

    action_count_eptotal[a][frame_num % max_steps] += 1
    for item in item_count_eptotal.keys():
        item_count_eptotal[item][frame_num %
                                 max_steps] += prev_state.inventory[prev_state.player_turn][item]

    # print action chosen by agent
    if env_render:
        state_tensor = torch.from_numpy(prev_state.getRepresentation(
            ar_obs=False)).float().to("cuda" if agent_params["dqn_config"].gpu >= 0 else "cpu").unsqueeze(0)
        q = agent.network.forward(
            state_tensor).cpu().detach().squeeze()
        probs = F.softmax(q.div(agent_params["softmax_temperature"]), dim=0)

        print_out(f"Action taken: {a} {Action(a)}")
        print_out(f"Action probabilities: {probs}")

    state, reward_list, outcome, episode_done, info = env.step(Action(a))
    # if it fails, outcome == [False, item wanted/location of craft]
    # if it succeed, outcome == [True, item gained/location of craft]

    if ar_obs:
        # check if this state is failure or no based on the inventory
        try:
            AR.perceive(prev_state, a, outcome, frame_num,
                        print_result=print_result)
        except ValueError as e:
            # example error msg
            # ValueError: array must not contain infs or NaNs
            readjust_ep(e)
            continue
        except OverflowError as e:
            # example error msg
            # OverflowError: Error in function ibeta_derivative<d>(%1%,%1%,%1%): Overflow Error
            readjust_ep(e)
            continue
        except RuntimeError as e:
            # example error msg
            # RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 600.
            readjust_ep(e)
            continue

    reward = reward_list[0]  # reward is list with length based on num_agents
    total_reward += reward

    if eval_running:
        steps_since_eval_began += 1
    else:
        steps_since_eval_ran += 1
        frame_num += 1

    # print results
    if env_render:
        print_out("Inventory")
        print_out(
            {key: val for key, val in state.inventory[0].items() if val > 0})
        print_out("")

        print_out("Agent")
        print_out(f"root folder \t attribute sets \t\t\t\t param")
        print_out(f"{ag_models_root} \t {attr_dic} \t {exp_param}")
        if ability:
            print_out("")
            print_out(f"player rating \t {state.ab_rating['player']}")
        print_out("")

        print_out(f"Timestep: {frame_num} \t Total reward: {total_reward}")
        print_out("---------------------------------------------")
        print_out("")

        # if frame_num % max_steps == 0:
        input()

    # handle episode done
    if episode_done:
        ep_num = math.floor(frame_num/max_steps)-1
        save_results = False
        if ep_num % 5 == 0:
            save_results = True

        if eval_running:  # This is only run during training
            # print('Evaluation time step: ' + str(steps_since_eval_began) +
            #       ', episode ended with score: ' + str(total_reward))
            eval_total_score += total_reward
            eval_total_episodes += 1
        else:
            score_str = ''

            average_total_reward = total_reward / num_trials

            if agent_params["test_mode"]:
                score_str = score_str + ', ' + agent.name + ": " + str(
                    total_reward) + " (" + "{:.2f}".format(average_total_reward) + ")"
            else:
                score_str = score_str + ', ' + \
                    agent.name + ": " + str(total_reward)

            if not env_render:  # not ar_obs # if ar is off then print as normal
                1  # print('Time step: ' + str(frame_num) +
                #       ', ep scores:' + score_str[1:])

            if agent_params["test_mode"]:
                with open(agent_params["log_dir"] + testing_scores_file, 'a') as fd:
                    fd.write("'" + float_to_str(seed) + ',' +
                             agent.name + ',' + str(total_reward) + '\n')

                # plot averages for each episode
                step_id = [x for x in range(max_steps)]
                print_out(f"DONE with episode {ep_num}")

                if ar_obs and "old_ar" not in exp_param and save_results:
                    fname = f'ep_{ep_num}'
                    ##################################################
                    # plot pdf max of ability
                    for x in step_id:
                        if len(pdf_max_ep[x]) == 0:
                            pdf_max_ep[x] = [AR.alpha_pdf_max_step[x]]
                        else:
                            pdf_max_ep[x].append(AR.alpha_pdf_max_step[x])

                    # plot pdf_max_ep of ability
                    # plt.boxplot(pdf_max_ep, positions=step_id)
                    # plt.xticks(
                    #     # all positions
                    #     ticks=np.arange(0, max_steps, 1),
                    #     labels=[
                    #         # blank out non-5 labels
                    #         str(x) if x % 5 == 0 else "" for x in range(max_steps)
                    #     ]
                    # )
                    # # plt.yticks(np.arange(0, 1+0.05, 0.05))
                    # # plt.yticks(np.arange(0, 1200, 100))
                    # plt.title(
                    #     "boxplot of alpha values with max pdf for each timestep averaged for all ep")
                    path = f'{AR.log_dir}/pdf_max_boxplot_ability'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    # plt.savefig(f'{path}/ep_{ep_num}')
                    # plt.clf()

                    data = np.asarray(pdf_max_ep)
                    np.savetxt(path+'/'+fname+".csv", data, delimiter=",")

                    # plot pdf max MAPE
                    # g_truth = list(attr_dic.values())[0]
                    # if "ability" in exp_param:
                    #     g_truth = ab_rating['player']
                    # for x in step_id:
                    #     mape = np.abs(
                    #         (g_truth - AR.alpha_pdf_max_step[x])/g_truth) * 100
                    #     if len(pdf_max_mape_ep[x]) == 0:
                    #         pdf_max_mape_ep[x] = [mape]
                    #     else:
                    #         pdf_max_mape_ep[x].append(mape)
                    #     if mape > max_mape:
                    #         max_mape = mape  # later used for the y_ticks

                    # plt.boxplot(pdf_max_mape_ep, positions=step_id)
                    # plt.xticks(
                    #     # all positions
                    #     ticks=np.arange(0, max_steps, 1),
                    #     labels=[
                    #         # blank out non-5 labels
                    #         str(x) if x % 5 == 0 else "" for x in range(max_steps)
                    #     ]
                    # )
                    # # plt.yticks(np.arange(0, max_mape+20, 20))
                    # plt.title(
                    #     "MAPE boxplot of alpha values with max pdf for each timestep averaged for all ep")
                    # path = f'{AR.log_dir}/pdf_max_mape_boxplot_ability'
                    # if not os.path.exists(path):
                    #     os.makedirs(path)
                    # plt.savefig(f'{path}/ep_{ep_num}')
                    # plt.clf()

                    # data = np.asarray(data)
                    # np.savetxt(path+fname+".csv", data, delimiter=",")

                    ##################################################
                    # plot preference (4 plots for 4 values)
                    for i in range(4):
                        for x in step_id:
                            if len(pdf_max_ep_pref[i][x]) == 0:
                                pdf_max_ep_pref[i][x] = [
                                    AR.alpha_pdf_max_step_pref[x][i]]
                            else:
                                pdf_max_ep_pref[i][x].append(
                                    AR.alpha_pdf_max_step_pref[x][i])

                        # plot pdf_max_ep of ability
                        # plt.boxplot(pdf_max_ep_pref[i], positions=step_id)
                        # plt.xticks(
                        #     # all positions
                        #     ticks=np.arange(0, max_steps, 1),
                        #     labels=[
                        #         # blank out non-5 labels
                        #         str(x) if x % 5 == 0 else "" for x in range(max_steps)
                        #     ]
                        # )
                        # # plt.yticks(np.arange(0, 1+0.05, 0.05))
                        # # plt.yticks(np.arange(0, 1200, 100))
                        # plt.title(
                        #     "boxplot of alpha values with max pdf for each timestep averaged for all ep")
                        path = f'{AR.log_dir}/pdf_max_boxplot_pref{i}'
                        if not os.path.exists(path):
                            os.makedirs(path)
                        # plt.savefig(f'{path}/ep_{ep_num}')
                        # plt.clf()
                        data = np.asarray(pdf_max_ep_pref[i])
                        np.savetxt(path+'/'+fname+".csv", data, delimiter=",")

        reset_all()

        # Model evaluation (only during Training)
        if not agent_params["test_mode"]:

            if frame_num >= eval_start_time and steps_since_eval_ran >= eval_freq:
                eval_running = True
                eval_total_score = 0
                eval_total_episodes = 0

                while steps_since_eval_ran >= eval_freq:
                    steps_since_eval_ran -= eval_freq

            elif steps_since_eval_began >= eval_steps:

                ave_eval_score = float(eval_total_score) / eval_total_episodes
                print_out('Evaluation ended with average score of ' +
                          str(ave_eval_score))

                with open(agent_params["log_dir"] + training_scores_file, 'a') as fd:
                    fd.write(str(agent.numSteps) + ',' +
                             str(ave_eval_score) + '\n')

                if ave_eval_score > best_eval_average:
                    best_eval_average = ave_eval_score
                    print_out('New best eval average of ' +
                              str(best_eval_average))
                    agent.save_model()
                else:
                    print_out('Did not beat best eval average of ' +
                              str(best_eval_average))

                eval_running = False
                steps_since_eval_began = 0

# get results
if ar_obs:
    AR.get_result(max_training_frames, attr_dic_to_str(
        attr_dic, inc_weight=True), ar_out_param, item_count_eptotal)
else:
    # copied from ar.generate_output()
    # TODO: divide this data processing into functions

    # Process the data
    avg_action_progression = {}
    for action, timestep_count in action_count_eptotal.items():
        avg_action_progression[action] = []
        total_count = 0
        for i in range(max_steps):
            avg_action_timestep_count = action_count_eptotal[action][i] / (
                max_training_frames/max_steps)
            avg_action_progression[action].append(
                total_count + avg_action_timestep_count)
            total_count += avg_action_timestep_count

    # Plot
    fig, ax = plt.subplots()

    for action, avg_counts in avg_action_progression.items():
        ax.plot(range(max_steps), avg_counts, label=Action(action).name)

    ax.set_ylim(0, 20)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average Progression Count")
    ax.set_title(
        f"Average Action Progression Over Time\n"
        f"Preference: {attr_dic_to_str(attr_dic, inc_weight=True)}\n"
        f"Ability Level: {ab_rating['player']}\nCrafting Difficulty: {ab_rating['craft']}"
    )
    ax.legend()

    # Save figure
    fig_path = real_path + result_folder + "/avg_action_progression/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        print_out(f"Created new folder: {fig_path}")

    if "pref" in exp_param:
        fig_path += f"{attr_dic_to_str(attr_dic, inc_weight=True)}"
    if "ability" in exp_param:
        fig_path += f"_{ab_rating['player']}"
    fig_path += ".jpg"

    plt.savefig(fig_path)
    plt.clf()

    # Aggregate average count per action
    avg_action_count = {action: sum(
        counts) / (max_training_frames/max_steps) for action, counts in action_count_eptotal.items()}

    # Convert to DataFrame
    df = pd.DataFrame(list(avg_action_count.items()),
                      columns=["Action", "Average Count"])
    df["Action"] = df["Action"].apply(
        lambda x: Action(x).name)  # if keys are enum members

    # Plot bar chart
    fig, ax = plt.subplots()
    fig.suptitle(
        f"Average Actions per Episode\nPreference: {attr_dic_to_str(attr_dic, inc_weight=True)}\n"
        f"Ability Level: {ab_rating['player']}\nCrafting Difficulty: {ab_rating['craft']}"
    )

    ax.set_ylim(0, 25)
    ax.bar(df["Action"], df["Average Count"], color=[
           COLORS[i % len(COLORS)] for i in range(len(df))])

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Count per Episode")
    plt.tight_layout()

    # Save figure
    fig_path = real_path + result_folder + "/avg_action_histogram/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        print_out(f"Created new folder: {fig_path}")

    if "pref" in exp_param:
        fig_path += f"{attr_dic_to_str(attr_dic, inc_weight=True)}"
    if "ability" in exp_param:
        fig_path += f"_{ab_rating['player']}"
    fig_path += ".jpg"

    plt.savefig(fig_path)
    plt.clf()

    # Count average items collected in each timestep
    avg_item_count = {k: [] for k in item_count_eptotal.keys()}
    ep_num = max_training_frames/max_steps
    for step in range(max_steps):
        for item in item_count_eptotal.keys():
            avg_item_count[item].append(item_count_eptotal[item][step]/ep_num)

    df = pd.DataFrame(avg_item_count)

    fig, ax = plt.subplots()
    fig.suptitle(
        f"Average Items Collected\nPreference: {attr_dic_to_str(attr_dic, inc_weight=True)}\nAbility Level: {ab_rating['player']}\nCrafting Difficulty: {ab_rating['craft']}")
    if "degrade" in exp_param:
        ax.set_ylim(0, 20)
    else:
        ax.set_ylim(0, 5)
    for i, column in enumerate(df.columns):
        label = column
        linewidth = 1
        if column in attr_dic:
            label += f" [{attr_dic[column]}]"
            linewidth += attr_dic[column] * 3
            if column in RECIPES:
                label += f"\n{RECIPES[column][0]}"
        ax.plot(df.index, df[column], label=label,
                color=COLORS[i], linewidth=linewidth)

    box = ax.get_position()
    # [left, bottom, width, height]
    ax.set_position([box.x0, 0.1, box.width * 0.8, box.height * 0.9])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save figure
    fig_path = real_path + result_folder + "/avg_item_count/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        print_out(f"Created new folder: {fig_path}")

    if "pref" in exp_param:
        fig_path += f"{attr_dic_to_str(attr_dic, inc_weight=True)}"
    if "ability" in exp_param:
        fig_path += f"_{ab_rating['player']}"
    fig_path += ".jpg"

    plt.savefig(fig_path)
    plt.clf()

# endregion
