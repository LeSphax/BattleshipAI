import time
import random

import numpy as np
import gym_battleship
import gym
from docopt import docopt

from random_player import RandomPlayer
from q_learning_player import QLearningPlayer

_USAGE = '''
Usage:
    main <player_mode> [--render]
        
Required:
<player_mode>      Use a random player or a q-learning player (values: random or ai)

Options:
    --render       Render the environment during training (This will greatly slow down training)     

'''
options = docopt(_USAGE)

player_mode = str(options['<player_mode>'])
render_training = options['--render']

env = gym.make('Battleship-v0')

ROWS = 5
COLUMNS = 5
BATCH_SIZE = 128
MAX_STEPS = 60000

if player_mode == 'ai':
    player = QLearningPlayer(ROWS, COLUMNS)
else:
    player = RandomPlayer(ROWS, COLUMNS)

episode_lengths = []
trajectories = {
    'boards': [],
    'actions': [],
    'rewards': []
}

turns = 0
board = env.reset()
done = False
player_turn = 0
last_summary_time = time.time()
for t in range(1, MAX_STEPS):
    turns += 1
    learning_rate = 0.01

    if t < 10000:
        epsilon = 0.5
    elif t < 20000:
        epsilon = 0.8
    else:
        epsilon = 0.95

    trajectories['boards'].append(board)
    action = player.act(board, epsilon)
    board, reward, done, _ = env.step(action)
    player_turn = (player_turn + 1) % 2
    if render_training:
        env.render()

    trajectories['actions'].append(np.asarray(action))
    trajectories['rewards'].append(reward)

    if t % BATCH_SIZE == 0:
        player.train(boards=trajectories['boards'], actions=trajectories['actions'], rewards=trajectories['rewards'],
                     batch_size=BATCH_SIZE, learning_rate=learning_rate)

        # for i in range(BATCH_SIZE):
        #     print(trajectories['boards'][i])
        #     print(trajectories['actions'][i])
        #     print(trajectories['rewards'][i])

        trajectories = {
            'boards': [],
            'actions': [],
            'rewards': []
        }
    if done:
        episode_lengths.append(turns)
        turns = 0
        board = env.reset()
        done = False
        player_turn = 0

    # print('Player {} has won the game after {} turns'.format(player_turn, turns))
    if t % 10000 == 0:
        print('Average episode length: {} turns'.format(np.mean(episode_lengths)))
        tps = np.sum(episode_lengths) / (time.time() - last_summary_time)
        print('Turns per second: {}'.format(tps))
        last_summary_time = time.time()
        episode_lengths = []

while True:
    action = player.act(board, 1)
    board, reward, done, _ = env.step(action)
    player_turn = (player_turn + 1) % 2
    env.render()
    time.sleep(0.3)

    if done:
        turns = 0
        board = env.reset()
        done = False
        player_turn = 0