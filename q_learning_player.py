import random
import tensorflow as tf
import numpy as np

class QLearningPlayer(object):

    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns

        self.board = tf.placeholder(shape=[None, rows, columns], dtype=tf.float32, name="board")
        self.actual_values = tf.placeholder(shape=[None], dtype=tf.float32, name="actual_value")
        self.chosen_actions = tf.placeholder(shape=[None, 2], dtype=tf.int32, name="chosen_action")
        self.batch_size = tf.placeholder(shape=(), dtype=tf.int32, name="batch_size")
        self.learning_rate = tf.placeholder(shape=(), dtype=tf.float32, name="learning_rate")

        input = tf.reshape(self.board, [-1, rows, columns, 1])

        layer = tf.contrib.layers.conv2d(
            inputs=input,
            num_outputs=8,
            kernel_size=3,
            padding="same",
            activation_fn=tf.nn.tanh,
            stride=1,
            weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
        )

        self.action_values = tf.contrib.layers.conv2d(
            inputs=layer,
            num_outputs=1,
            kernel_size=3,
            padding="same",
            activation_fn=tf.nn.tanh,
            stride=1,
            weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
        )

        # Get the predictions for the chosen actions only
        batch_indices = tf.reshape(tf.range(self.batch_size), [-1, 1])
        gather_indices = tf.concat([batch_indices, self.chosen_actions], 1)
        self.chosen_action_values = tf.gather_nd(self.action_values, gather_indices)

        self.chosen_action_values = tf.reshape(self.chosen_action_values, [-1])

        # Calculate the loss
        self.losses = tf.squared_difference(self.actual_values, self.chosen_action_values)
        self.loss = tf.reduce_mean(self.losses)
        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.action_values),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.action_values))
        ])

        self.sess = tf.Session()
        self.sess.__enter__()
        self.sess.run(tf.initializers.global_variables())

    def act(self, board, epsilon):
        action_values = self.sess.run(self.action_values, {self.board: [board]})
        flattened_array = np.reshape(action_values, [-1])

        # Epsilon greedy choice
        rand = random.random()
        if rand < epsilon:
            idx = np.argmax(flattened_array)
        else:
            idx = random.randint(0, len(flattened_array) - 1)

        return idx // self.columns, idx % self.columns

    def train(self, boards, actions, rewards, batch_size, learning_rate):
        _, values, chosen_values, chosen_actions, losses, rewards = self.sess.run([self.train_op, self.action_values, self.chosen_action_values, self.chosen_actions, self.losses, self.actual_values],   {
            self.board: boards,
            self.chosen_actions: actions,
            self.actual_values: rewards,
            self.batch_size: batch_size,
            self.learning_rate: learning_rate
        })
        # for idx, value in enumerate(values):
        #
        #     print(value)
        #     print(chosen_values[idx])
        #     print(chosen_actions[idx])
        #     print(np.shape(chosen_values))
        #     print(np.shape(losses))
        #     print(np.shape(rewards))
