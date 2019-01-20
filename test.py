import tensorflow as tf
import numpy as np

board = tf.placeholder(shape=[None, 3, 3], dtype=tf.int32, name="board")
chosen_actions = tf.placeholder(dtype=tf.int32, name="chosen_actions")
batch_size = tf.placeholder(dtype=tf.int32, name="batch_size")

batch_indices = tf.reshape(tf.range(batch_size), [-1, 1])
gather_indices = tf.concat([batch_indices, chosen_actions], 1)
action_predictions = tf.gather_nd(board, gather_indices)


indice_action = tf.argmax(tf.reshape(board, [-1]))

sess = tf.Session()
sess.__enter__()
print(np.shape([[0,0]]))

print(sess.run(action_predictions, {
    chosen_actions: np.array([[0,0], [2,1]]),
    board: np.array([[[1,2,3], [4,5,6], [7,8,9]], [[11,22,33], [44,55,66], [77,88,99]]]),
    batch_size:2
}))

print(sess.run(tf.reshape(board, [3,3]), {
    chosen_actions: np.array([[0,0], [2,1]]),
    board: np.array([[[1,2222,3], [4,5,6], [777,8,9]]]),
    batch_size:2
}))

print(sess.run(indice_action, {
    chosen_actions: np.array([[0,0], [2,1]]),
    board: np.array([[[1,2,3], [4,5,6], [777,8,9]]]),
    batch_size:2
}))