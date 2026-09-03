scenarios = {}


# Defaults
scenarios["default"] = {}

# by default ingredients won't regenerate once its collected
scenarios["default"]["regeneration"] = False

scenarios["default"]["starting_items"] = [{}, {}]

scenarios["default"]["num_spawned"] = {
    # default
    "wood": 4,
    "iron": 4,
    "grass": 4,
    "gem": 2,
    "gold": 2,
    "workbench": 2,
    "toolshed": 2,
    "factory": 2,
    # "wood": 4,
    # "iron": 4,
    # "grass": 0,
    # "gem": 12,
    # "gold": 12,
    # "workbench": 0,
    # "toolshed": 2,
    # "factory": 2,
}

scenarios["default"]["hidden_items"] = [
    ["wood", "grass"]
]

# RL training/testing scenario

# region WEIGHT
# Goal preferences
# {"gem": 0.1, "gold": 0.9}
# {"gem": 0.3, "gold": 0.7}
# {"gem": 0.5, "gold": 0.5}
# {"gem": 0.7, "gold": 0.3}
# {"gem": 0.9, "gold": 0.1}

# Belief
# {"iron": 0.7, "wood": -1, "grass": 1}
# {"iron": 0.7, "wood": 1, "grass": -1}

# Skill: {"axe": 1, "bridge": 0.8}

# Modified Q-Function: {"cloth": 0, "stick": 0}
# endregion

scenarios["train"] = {}

scenarios["train"]["attr_sets"] = [
    # {"gem": 1, "gold": 1}
    # {"plank": 1, "stick": 0.5}
    # {"axe": 0.2, "bridge": 0.2, "gem": 1, "gold": 1}
    # {"axe": 1, "bridge": 1, "plank": 1, "stick": 1}
    {"axe": 1, "bridge": 1, "plank": 1}
    # {"cloth": 1, "stick": 1, "plank": 1, "rope": 1}
    # {"cloth": 0.1, "stick": 0.1, "plank": 0.1, "rope": 0.7}
]

scenarios["eval"] = {}
scenarios["eval"]["attr_sets"] = [
    # {"gem": 1, "gold": 1}
    # {"plank": 1, "stick": 0.5}
    # {"axe": 1, "bridge": 0.7}
    {"cloth": 0.1, "stick": 0.1, "plank": 0.1, "rope": 0.7}
    # {"axe": 0.7, "bridge": 0.1, "plank": 0.1, "stick": 0.1}
    # {"axe": 0.1, "bridge": 0.8, "plank": 0.1}
    # {"cloth": 0.1, "stick": 0.9}
]
