from constants import DEFAULT_PARAMS


class Screen():

    def __init__(self, size):
        self.size = size
        self.rows = []
        for y in range(0, self.size[1]):
            self.rows.append('.' * self.size[0])

    def add_sprite(self, sprite_pos, sprite):
        row = len(self.rows) - sprite_pos.y - 1
        self.rows[row] = self.rows[row][:sprite_pos.x] + \
            sprite + self.rows[row][sprite_pos.x + 1:]

    def render(self, filename=None):
        for y in range(0, self.size[1]):
            print_or_log(self.rows[y], filename)


def print_or_log(str, filename):

    if filename is not None:
        with open(filename, 'a') as fd:
            fd.write(str + "\n")
    else:
        print(str)


def elo(ra, rb):
    # the probability of a winning over b
    # following elo rating formula
    return 1/(1 + 10 ** ((rb - ra)/400))


def is_default_param(param):
    if param in DEFAULT_PARAMS:
        return True
    return False
