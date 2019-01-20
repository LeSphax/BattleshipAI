import gym
import random
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering

#Board cell values
IDLE = 0
MISSED = 1
TOUCHED = 2
SUNK = 3
SHIP = 4

# Colors
BLACK = (0, 0, 0, 1)
BLUE = (0, 0, 1, 1)
GREEN = (0, 1, 0, 1)
RED = (1, 0, 0, 1)
PURPLE = (1, 0, 1, 1)
YELLOW = (0, 1, 1, 1)


class BattleshipEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.rows = 5
        self.columns = 5
        self.ship_size = 3

        self.observation_space = spaces.Box(low=0, high=3, shape=(10,), dtype=np.int32)
        self.action_space = spaces.Tuple((spaces.Discrete(self.rows), spaces.Discrete(self.columns)))
        self.boards = None

        self.viewer = None
        self.colors = None

        self.player_turn = None
        self.tiles = None

    def seed(self, seed):
        random.seed(seed)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.last_action = action
        reward = 0

        target = self.boards[self.player_turn][action[0]][action[1]]

        if target == IDLE:
            self.boards[self.player_turn][action[0]][action[1]] = MISSED
        elif target == SHIP:
            reward = 1
            self.boards[self.player_turn][action[0]][action[1]] = TOUCHED

        self.player_turn = (self.player_turn + 1) % 2

        done = False
        if self.check_win():
            done = True

        return self.hide_ships(self.player_turn), reward, done, {}

    def check_win(self):
        for player_idx in range(2):
            if SHIP not in self.boards[player_idx]:
                return True
        return False

    def reset(self):
        self.player_turn = 0
        self.boards = np.zeros((2, self.rows, self.columns))
        for player_idx in range(2):
            ship_start_position = [random.randint(0, self.rows - 1), random.randint(0, self.columns - 1)]

            ship_direction = -1
            while ship_direction == -1 \
                    or ship_direction == 0 and self.rows - ship_start_position[0] < self.ship_size \
                    or ship_direction == 1 and ship_start_position[0] < self.ship_size - 1 \
                    or ship_direction == 2 and self.rows - ship_start_position[1] < self.ship_size \
                    or ship_direction == 3 and ship_start_position[1] < self.ship_size - 1:
                ship_direction = random.randint(0, 3)

            x = 0
            y = 0
            if ship_direction == 0:
                x = 1
            elif ship_direction == 1:
                x = -1
            elif ship_direction == 2:
                y = 1
            else:
                y = -1

            for ship_idx in range(self.ship_size):
                self.boards[player_idx] \
                    [ship_start_position[0] + x * ship_idx] \
                    [ship_start_position[1] + y * ship_idx] = SHIP

        return self.hide_ships(self.player_turn)

    def hide_ships(self, player_idx):
        visible_board = np.copy(self.boards[player_idx])
        visible_board[visible_board == SHIP] = IDLE
        return visible_board

    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 400

        tile_w = 10.0
        tile_h = 10.0
        spacing = 3.0

        y_offset = 100

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.tiles = np.empty((2, self.rows, self.columns), dtype=object)
            self.colors = np.empty((2, self.rows, self.columns), dtype=object)

            for player_idx in range(2):
                x_offset = 100 + 200 * player_idx
                for row in range(self.rows):
                    for column in range(self.columns):
                        l = x_offset + (spacing + tile_w) * column
                        r = l + tile_w
                        b = y_offset + (spacing + tile_h) * row
                        t = b + tile_h
                        tile = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                        self.tiles[player_idx][row][column] = tile
                        self.colors[player_idx][row][column] = tile.attrs[0]
                        self.viewer.add_geom(tile)

        for player_idx in range(2):
            for row in range(self.rows):
                for column in range(self.columns):
                    color = None
                    cell = self.boards[player_idx][row][column]
                    if cell == IDLE:
                        color = BLACK
                    elif cell == MISSED:
                        color = BLUE
                    elif cell == TOUCHED:
                        color = PURPLE
                    elif cell == SUNK:
                        color = RED
                    elif cell == SHIP:
                        color = GREEN
                    self.colors[player_idx][row][column].vec4 = color

        self.colors[(self.player_turn + 1) % 2][self.last_action[0]][self.last_action[1]].vec4 = YELLOW

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
