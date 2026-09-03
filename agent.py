from abc import abstractmethod
from environment import CraftWorldState

# Abstract class to allow different types of Agent object
# Currently there's only one type of Agent: neural_q_learner.py


class Agent(object):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def perceive(self, reward: float, state: CraftWorldState, terminal: bool, is_eval: bool, model_file: str):
        pass

    def reset(self, attr_set, externally_visible_attr_sets):
        # this current setup assumes that attr is agent specific
        self.attr_set = attr_set
        self.externally_visible_attr_sets = externally_visible_attr_sets

        self.attribute = list(self.attr_set.keys())[0]
        for i in range(1, len(list(self.attr_set.keys()))):
            self.attribute = self.attribute + "_and_" + \
                list(self.attr_set.keys())[i]
