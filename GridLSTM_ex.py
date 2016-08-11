import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
# from tensorflow.contrib.rnn.python.ops import rnn_cell
from tensorflow.contrib.grid_rnn.python.ops import grid_rnn_cell
import numpy as np

#define parameters
n_input_x = 10
n_input_y = 10
n_input_z = 10

n_hidden = 128
n_classes = 2

x = tf.placeholder("float", [n_input_x, n_input_y, n_input_z])
y = tf.placeholder("float", [n_input_x, n_input_y, n_input_z, n_classes])

#generate random data
input_data = np.random.rand(n_input_x, n_input_y, n_input_z)
ground_truth = np.random.rand(n_input_x, n_input_y, n_input_z, n_classes)

#build GridLSTM
def GridLSTM_network(x):
    import pdb
    pdb.set_trace()

    x = tf.reshape(x, [-1,n_input_x])
    x = tf.split(0, n_input_y * n_input_z, x)
    lstm_cell = grid_rnn_cell.GridRNNCell(n_hidden, num_dims=3)
    # lstm_cell = rnn_cell.GridLSTMCell(n_hidden, use_peepholes=True)
    lstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * n_hidden, state_is_tuple=True)
    lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.8)

    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    return outputs

#initialize network, cost, optimizer and all variables
pred = GridLSTM_network(x)
pred = tf.pack(pred)
pred = tf.transpose(pred, [1,0,2,3])
temp_pred = tf.reshape(pred, [-1,n_classes])
temp_y = tf.reshape(y,[-1, n_classes])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(temp_pred, temp_y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(0,tf.cast(tf.sub(tf.nn.softmax(temp_pred),temp_y), tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while 1:
        print step
        step = step + 1
        # pdb.set_trace
        sess.run(optimizer, feed_dict={x: input_data, y: ground_truth})
