import math
import gym
import numpy as np
import random
from time import sleep
from vector_2d import Vector2D

from constants import Action, RECIPES, RAW_RECIPES, REWARDABLE_ITEMS, SPRITES, UNLIMITED_INV, LIMITED_INV

from utils import Screen, print_or_log, elo

_num_spawned = None
_reward = []


class CraftWorldState():

    def __init__(self, size, action_space, n_agents=1, ingredient_regen=True, max_steps=300, hidden_items=[], exp_param=[], ab_rating=[], test_mode=False):
        self.player_turn = 0
        self.action_space = action_space
        self.n_agents = n_agents
        self.ingredient_regen = ingredient_regen
        self.size = size
        self.max_steps = max_steps
        self.test_mode = test_mode

        self.ab_rating = ab_rating
        self.exp_param = exp_param

        self.reset()

        self.hidden_items = hidden_items

        self._max_inventory = UNLIMITED_INV

        if "limit" in exp_param:
            self._max_inventory = LIMITED_INV

    def step(self, action, assumed_reward_func=_reward):

        reward = [0] * self.n_agents
        outcome = [True, '']

        if self.terminal:
            return reward, outcome

        if action == Action.UP:
            if (self.objects["player"][self.player_turn].y + 1) < self.size[1]:
                self.objects["player"][self.player_turn].y += 1
        elif action == Action.DOWN:
            if (self.objects["player"][self.player_turn].y - 1) >= 0:
                self.objects["player"][self.player_turn].y -= 1
        elif action == Action.LEFT:
            if (self.objects["player"][self.player_turn].x - 1) >= 0:
                self.objects["player"][self.player_turn].x -= 1
        elif action == Action.RIGHT:
            if (self.objects["player"][self.player_turn].x + 1) < self.size[0]:
                self.objects["player"][self.player_turn].x += 1

        # Check if we can pick up an item
        if action == Action.COLLECT:
            for k in self._max_inventory.keys():

                can_pick_up = True

                if k == "gem" and self.inventory[self.player_turn]["axe"] == 0:
                    can_pick_up = False

                if k == "gold" and self.inventory[self.player_turn]["bridge"] == 0:
                    can_pick_up = False

                if not can_pick_up:
                    continue

                if self.inventory[self.player_turn][k] < self._max_inventory[k]:
                    for pos in self.objects[k]:
                        if self.objects["player"][self.player_turn].x == pos.x and self.objects["player"][self.player_turn].y == pos.y:
                            self.inventory[self.player_turn][k] += 1
                            reward[self.player_turn] += assumed_reward_func[self.player_turn][k]
                            self.objects[k].remove(pos)
                            if self.ingredient_regen:
                                self.objects[k].append(self.get_free_square())
                            if "degrade" in self.exp_param:
                                collect_prob = elo(
                                    self.ab_rating['player'], self.ab_rating['collect'])
                                if k == "gem" and self.inventory[self.player_turn]["axe"] > 0:
                                    # degrade axe
                                    self.inventory[self.player_turn]["axe"] *= collect_prob
                                    if self.inventory[self.player_turn]["axe"] < 0.1:
                                        self.inventory[self.player_turn]["axe"] = 0
                                if k == "gold" and self.inventory[self.player_turn]["bridge"] > 0:
                                    # degrade bridge
                                    self.inventory[self.player_turn]["bridge"] *= collect_prob
                                    if self.inventory[self.player_turn]["bridge"] < 0.1:
                                        self.inventory[self.player_turn]["bridge"] = 0
                            break

        # Check if we can craft
        if action == Action.CRAFT:
            for k, v in RECIPES.items():
                # if player location is at a crafting location
                if self.objects["player"][self.player_turn] in self.objects[v[0]]:
                    recipe_met = True

                    # check if player's inventory contains required ingredient
                    for ingredient, required_count in v[1].items():
                        if self.inventory[self.player_turn][ingredient] < required_count:
                            recipe_met = False
                            break

                    if recipe_met:
                        # get crafting probabilities
                        r_craft = random.random()
                        craft_prob = 1
                        if len(self.ab_rating) > 0 and "degrade" not in self.exp_param:
                            if "craft_loc" in self.exp_param:
                                # different crafting difficulty based on location
                                craft_prob = elo(
                                    self.ab_rating['player'], self.ab_rating['craft'][v[0]])
                                outcome[1] = v[0]
                            elif "craft_item" in self.exp_param:
                                if k not in self.ab_rating['craft']:
                                    craft_prob = 1
                                else:
                                    craft_prob = elo(
                                        self.ab_rating['player'], self.ab_rating['craft'][k])
                                outcome[1] = k
                            else:
                                # same crafting difficulty
                                craft_prob = elo(
                                    self.ab_rating['player'], self.ab_rating['craft'])

                            # self.ab_rating['player'], assumed_reward_func[self.player_turn][k] * self.ab_rating['craft'])  # different crafting difficulty

                        # craft using items in inventory
                        for ingredient, required_count in v[1].items():
                            self.inventory[self.player_turn][ingredient] -= required_count

                        # failed crafting, scatter collected items on the ground
                        if r_craft > craft_prob:
                            outcome[0] = False
                            for ingredient, required_count in v[1].items():
                                if ingredient in self._max_inventory:  # raw items
                                    # to check if it's ingredients collected from the ground (i.e. wood, iron)
                                    # this will keep crafted ingredients (i.e. plank, stick) in inventory
                                    for x in range(required_count):
                                        self.objects[ingredient].append(
                                            self.get_free_square())
                                elif ingredient in RAW_RECIPES:  # non raw items
                                    # hard coded raw recipes to convert ingreds back to raw
                                    ingred_recipe = RAW_RECIPES[ingredient]
                                    for raw_item in ingred_recipe:
                                        required_count = ingred_recipe[raw_item]
                                        for x in range(required_count):
                                            self.objects[raw_item].append(
                                                self.get_free_square())
                            if "fail_neg" in self.exp_param:
                                # Negative reward when agent failed to craft
                                reward[self.player_turn] -= 0.2
                            break

                        # update inventory when craft succeeds
                        self.inventory[self.player_turn][k] += 1
                        reward[self.player_turn] += assumed_reward_func[self.player_turn][k]

        self.steps += 1
        if self.steps >= self.max_steps:
            self.terminal = True

        # set player for multiagent environment
        self.player_turn = (self.player_turn + 1) % self.n_agents

        return reward, outcome

    def get_object_type_at_square(self, square):

        if square is None:
            return "none"

        for object_type, position_list in self.objects.items():
            for pos in position_list:
                if square.x == pos.x and square.y == pos.y:
                    return object_type

        return "none"

    def is_square_free(self, square):
        for position_list in self.objects.values():
            for pos in position_list:
                if square.x == pos.x and square.y == pos.y:
                    return False
        return True

    def get_free_square(self):
        square = Vector2D(random.randrange(
            0, self.size[0]), random.randrange(0, self.size[1]))
        while not self.is_square_free(square):
            square = Vector2D(random.randrange(
                0, self.size[0]), random.randrange(0, self.size[1]))
        return square

    def getNearestObjects(self, object_name, n=1):
        p_pos = self.objects["player"][self.player_turn]
        return sorted(self.objects[object_name], key=lambda x: p_pos.distance_to(x))[0:n]

    def getObjectCount(self, object_name):
        return len(self.objects[object_name])

    def reset(self):
        self.objects = {}
        # place players first
        if 'player' in _num_spawned:
            self.objects['player'] = []
            for i in range(_num_spawned['player']):
                self.objects['player'].append(self.get_free_square())

        # place the rest of objects
        keys = list(_num_spawned.keys())
        # make sure random.seed(...) has been called already for reproducibility
        random.shuffle(keys)
        for k in keys:
            if k == 'player':
                continue
            self.objects[k] = []
            for i in range(_num_spawned[k]):
                self.objects[k].append(self.get_free_square())

        # for k, v in _num_spawned.items():
        #     self.objects[k] = []
        #     for i in range(0, v):
        #         self.objects[k].append(self.get_free_square())

        self.inventory = []
        for _ in range(0, self.n_agents):
            agent_inv = {}
            for item in REWARDABLE_ITEMS:
                agent_inv[item] = 0
                # if "degrade" in self.exp_param and (item == "axe" or item == "bridge"):
                # agent_inv[item] = 1

            self.inventory.append(agent_inv)

        self.terminal = False
        self.steps = 0

    def getRepresentation(self, ar_obs=False, ar_param=[]):
        # ADD objects on the environment
        rep = []
        max_dist = np.sqrt(
            self.size[0] * self.size[0] + self.size[1] * self.size[1])
        p_pos = self.objects["player"][self.player_turn]
        angle_increment = 45  # Note: Should divide perfectly into 360

        sorted_keys = sorted(self.objects.keys())
        if ("belief" in self.exp_param and not ar_obs):
            sorted_keys = [
                x for x in sorted_keys if x not in self.hidden_items]
            sorted_keys.append("hidden")
        if "belief" in ar_param:
            sorted_keys = [
                x for x in sorted_keys if x not in ar_param['belief']]
            sorted_keys.append("hidden")

        for k in sorted_keys:
            if k != "player":

                # Sort by distance to the player so that nearer objects are represented earlier.
                if k == "hidden":
                    sorted_objects = []
                    for obj in self.hidden_items:
                        sorted_objects += self.objects[obj]
                    sorted_objects = sorted(
                        sorted_objects, key=lambda x: p_pos.distance_to(x))
                else:
                    sorted_objects = sorted(
                        self.objects[k], key=lambda x: p_pos.distance_to(x))

                for l_bound in range(-180, 180, angle_increment):
                    u_bound = l_bound + angle_increment
                    obj_rep = 0.0
                    for obj in sorted_objects:

                        # If we're right on top of the object, represent it as a 1 in all directions
                        if obj == p_pos:
                            obj_rep = 1.0
                        else:
                            angle = int(math.degrees(math.atan2(
                                obj.y - p_pos.y, obj.x - p_pos.x)))
                            if (angle >= l_bound and angle <= u_bound) or (angle - 360) == l_bound or (angle + 360) == u_bound:
                                obj_rep = 1.0 - \
                                    p_pos.distance_to(obj) / max_dist
                                break

                    rep.append(obj_rep)

        # ADD agent's inventory
        # Note: If recipes are added where required_count > 1, this will logic will need to be modified.
        sorted_keys = sorted(self.inventory[self.player_turn].keys())
        for k in sorted_keys:
            rep.append(min(self.inventory[self.player_turn][k], 1))

        # ADD agent's step count
        rep.append(self.steps / self.max_steps)

        # ADD reward for preference -- for executing agent
        # COMMENT OUT "and not self.test_mode" to use UVFA for an executing agent
        # and not self.test_mode:
        if "uvfa" in self.exp_param and "pref" in self.exp_param and not ar_obs:
            # if ("uvfa" in self.exp_param or "multivar" in self.exp_param) and "pref" in self.exp_param and not ar_obs:
            for item in sorted_keys:
                if _reward != []:
                    rep.append(_reward[self.player_turn][item])
                else:
                    rep.append(0)

        # ADD reward for preference -- for observing agent
        if "uvfa" in ar_param and "pref" in ar_param and "multivar" not in ar_param:
            for item in sorted_keys:
                if item in ar_param["uvfa"]:
                    rep.append(ar_param["uvfa"][item])
                else:
                    rep.append(0)

        if "uvfa" in ar_param and "multivar" in ar_param:
            if "pref" in ar_param["multivar"]:
                for item in sorted_keys:
                    if item in ar_param["multivar"]["pref"]:
                        rep.append(ar_param["multivar"]["pref"][item])
                    else:
                        rep.append(0)

        # ADD ability level
        if "uvfa" in self.exp_param and "ability" in self.exp_param and not ar_obs:
            # if ("uvfa" in self.exp_param or "multivar" in self.exp_param) and "ability" in self.exp_param and not ar_obs:
            if "elo_lens" in self.exp_param:
                craft_prob = self.elo_lens(p_pos)
                rep.append(craft_prob)
            # scale rating
            scaled_rating = (self.ab_rating['player'] - self.ab_rating['min']) / (
                self.ab_rating['max'] - self.ab_rating['min'])
            rep.append(scaled_rating)

        # ADD ability level -- for observing agent
        if "uvfa" in ar_param and "ability" in ar_param and "multivar" not in ar_param:
            if "elo_lens" in ar_param:
                craft_prob = self.elo_lens(p_pos)
                rep.append(craft_prob)
            # scale rating
            scaled_rating = (ar_param["uvfa"] - self.ab_rating['min']) / (
                self.ab_rating['max'] - self.ab_rating['min'])
            rep.append(scaled_rating)

        if "uvfa" in ar_param and "multivar" in ar_param:
            if "elo_lens" in ar_param:
                craft_prob = self.elo_lens(p_pos)
                rep.append(craft_prob)
            if "ability" in ar_param["multivar"]:
                # scale rating
                scaled_rating = (ar_param["multivar"]["ability"] - self.ab_rating['min']) / (
                    self.ab_rating['max'] - self.ab_rating['min'])
                rep.append(scaled_rating)

        return np.array(rep, dtype=np.float32)

    def render(self, use_delay=False, log_dir=None):

        if log_dir is not None:
            filename = log_dir + 'state_log.txt'
        else:
            filename = None

        print_or_log("", filename)

        sorted_keys = sorted(self.inventory[0].keys())
        for k in sorted_keys:
            inv_str = k + ":"
            for agent_num in range(0, self.n_agents):
                inv_str = inv_str + " " + str(self.inventory[agent_num][k])
            if filename:
                print_or_log(inv_str, filename)

        print_or_log("", filename)

        screen = Screen(self.size)

        for k, v in self.objects.items():
            if k == "player":
                for agent_num in range(0, self.n_agents):
                    screen.add_sprite(v[agent_num], str(agent_num))
            else:
                for pos in v:
                    # get the key values
                    screen.add_sprite(pos, SPRITES[k])

        screen.render(filename)

        print()

        if use_delay:
            sleep(0.1)

    def elo_lens(self, player_pos):
        craft_prob = 0
        for k, v in RECIPES.items():
            # if player location is at a crafting location
            if player_pos in self.objects[v[0]]:
                recipe_met = True

                # check if player's inventory contains required ingredient
                for ingredient, required_count in v[1].items():
                    if self.inventory[self.player_turn][ingredient] < required_count:
                        recipe_met = False
                        break

                if recipe_met:
                    craft_prob = 1
                    # get crafting probabilities
                    if len(self.ab_rating) > 0 and "degrade" not in self.exp_param:
                        if "craft_loc" in self.exp_param:
                            # different crafting difficulty based on location
                            craft_prob = elo(
                                self.ab_rating['player'], self.ab_rating['craft'][v[0]])
                        elif "craft_item" in self.exp_param:
                            if k not in self.ab_rating['craft']:
                                craft_prob = 1
                            else:
                                craft_prob = elo(
                                    self.ab_rating['player'], self.ab_rating['craft'][k])
                        else:
                            # same crafting difficulty
                            craft_prob = elo(
                                self.ab_rating['player'], self.ab_rating['craft'])
        return craft_prob


class CraftWorld(gym.Env):

    def __init__(self, scenario, size=(10, 10), n_agents=1, allow_no_op=False, render=False, ingredient_regen=True, max_steps=300, exp_param=[], ab_rating=[], test_mode=False):

        global _num_spawned
        _num_spawned = scenario["num_spawned"]
        _num_spawned["player"] = n_agents

        self.allow_no_op = allow_no_op
        self.render = render
        self.ingredient_regen = ingredient_regen

        if self.allow_no_op:
            self.action_space = gym.spaces.Discrete(7)
        else:
            self.action_space = gym.spaces.Discrete(6)

        hidden_items = []
        if "belief" in exp_param:
            hidden_items = scenario["hidden_items"][0]

        self.state = CraftWorldState(
            size, self.action_space, n_agents=n_agents, ingredient_regen=ingredient_regen, max_steps=max_steps, hidden_items=hidden_items, exp_param=exp_param, ab_rating=ab_rating, test_mode=test_mode)

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=self.state.getRepresentation().shape, dtype=np.float32)

        print("Craftworld Init -- Observation space shape:",
              self.observation_space.shape)

        self.exp_param = exp_param
        self.test_mode = test_mode
        self.ab_rating = ab_rating

    # This 'step' is defined just to meet the requirements of gym.Env.
    # It returns a numpy array representation of the state based on an 'eye' encoding.
    def step(self, action):
        reward, outcome = self.state.step(action)
        info = {}

        if self.render:
            self.state.render(True)

        ag_state = self.state

        return ag_state, reward, outcome, self.state.terminal, info

    def reset(self, agents, seed):

        # Reset the reward function
        _reward.clear()
        for agent in agents:
            reward_dic = {}

            # to set the actual reward when an item is collected
            if "uvfa" in self.exp_param:
                # get randomized number sum to 1 for uvfa rewards during training
                # only for inferring preference
                if not self.test_mode and "pref" in self.exp_param:
                    reward_list = np.random.dirichlet(
                        np.ones(len(agent.attr_set.keys())))
                    # VERY HACKY FLAG :) so that state can know which one has uvfa
                else:
                    reward_list = [agent.attr_set[x]
                                   for x in agent.attr_set.keys()]
                reward_dic["uvfa"] = 0

            # for all attributes
            for item in REWARDABLE_ITEMS:
                if item in agent.attr_set.keys():
                    if "uvfa" in self.exp_param and not self.test_mode:
                        item_id = list(agent.attr_set.keys()).index(item)
                        reward_dic[item] = reward_list[item_id]
                    else:
                        reward_dic[item] = agent.attr_set[item]
                else:
                    reward_dic[item] = 0
                    if "incentive" in self.exp_param:
                        # Incentivize collecting req items to craft attribute set
                        for k, v in RECIPES.items():
                            if item in v[1] and k in agent.attr_set:
                                reward_dic[item] = 0.2
                                break

            _reward.append(reward_dic)

        # reset seed
        random.seed(seed)

        # reset state
        self.state.reset()

        if self.render:
            self.state.render(True)

        return self.state
