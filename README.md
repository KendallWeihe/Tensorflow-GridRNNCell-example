# Tensorflow-GridRNNCell-example
Example code for how a GridRNNCell functions

This program is really just for anyone having syntax issues with the `GridRNNCell` in Tensorflow's implementation

There are currently two implementations of the Grid LSTM network in Tensorflow. One is in `tensorflow.contrib.rnn.python.ops rnn_cell.py` and the other is in `tensorflow.contrib.grid_rnn`. I recommend using the latter implementation. Mainly for the reason that it implements N-dimensional Gird LSTM networks (one of the main features from the original [paper](http://arxiv.org/abs/1507.01526)), whereas the former implementation does not. 
