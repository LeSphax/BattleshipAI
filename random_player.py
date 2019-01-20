import random


class RandomPlayer(object):

    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns

    def act(self, board, epsilon):
        return random.randint(0, self.rows - 1), random.randint(0, self.columns - 1)

    def train(self, boards, actions, rewards, batch_size, learning_rate):
        pass
