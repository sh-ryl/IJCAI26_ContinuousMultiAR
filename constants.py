import enum


class Action(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    COLLECT = 4
    CRAFT = 5
    NO_OP = 6


SPRITES = {
    "wood": '|',
    "iron": '=',
    "grass": '#',
    "gem": '@',
    "gold": '*',
    "workbench": 'W',
    "toolshed": 'T',
    "factory": 'F'
}

REWARDABLE_ITEMS = [
    "axe",
    "bed",
    "bridge",
    "cloth",
    "gem",
    "gold",
    "grass",
    "iron",
    "plank",
    "rope",
    "stick",
    "wood"
]

RECIPES = {
    # originally axe requires iron 1 and stick 1, bridge requires iron 1 wood 1
    "axe": ("toolshed", {"iron": 1, "stick": 1}),
    "bed": ("workbench", {"grass": 1, "plank": 1}),
    "bridge": ("factory", {"iron": 1, "wood": 1}),
    "cloth": ("factory", {"grass": 1}),
    "plank": ("toolshed", {"wood": 1}),
    "rope": ("toolshed", {"grass": 1}),
    "stick": ("workbench", {"wood": 1}),

    # degrade
    # "axe": ("toolshed", {"iron": 1, "wood": 2}),
    # "bed": ("workbench", {"grass": 1, "plank": 1}),
    # "bridge": ("factory", {"iron": 2, "wood": 1}),
    # "cloth": ("factory", {"grass": 1}),
    # "plank": ("toolshed", {"wood": 1}),
    # "rope": ("toolshed", {"grass": 1}),
    # "stick": ("workbench", {"wood": 1})
}

RAW_RECIPES = {
    "axe": {"iron": 1, "wood": 1},
    "bed": {"grass": 1, "wood": 1},
    "bridge": {"iron": 1, "wood": 1},
    "cloth": {"grass": 1},
    "plank": {"wood": 1},
    "rope": {"grass": 1},
    "stick": {"wood": 1}
}

UNLIMITED_INV = {"wood": 999,
                 "iron": 999,
                 "grass": 999,
                 "gem": 999,
                 "gold": 999
                 }

LIMITED_INV = {"wood": 1,
               "iron": 1,
               "grass": 1,
               "gem": 999,
               "gold": 999
               }

# params that will show up in some of the code
DEFAULT_PARAMS = ["co", "train", "eval",
                  "AR", "render", "limit",
                  "uvfa", "multivar",
                  "belief", "ability", "pref",
                  "modbi",
                  "incentive", "fail_neg",
                  "degrade",
                  "old_ar",
                  "craft_loc", "craft_item",
                  "elo_lens"
                  ]

# INCENTIVE the environment will give 0.2 reward for collecting items required to craft
# (e.g. for plank, it requires wood. if agent collect wood, agent will get 0.2 reward)

# DEGRADE the environment will degrade axe and bridges after collecting items

COLORS = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
    '#aec7e8',  # light blue
    '#ffbb78',  # light orange
    '#98df8a',  # mint green
    '#ff9896',  # salmon pink
    '#c5b0d5',  # lavender
    '#c49c94',  # rose taupe
    '#f7b6d2',  # pink blush
    '#c7c7c7',  # silver gray
    '#dbdb8d',  # pale olive
    '#9edae5',  # pale cyan
]
