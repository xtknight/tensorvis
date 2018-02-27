"""
Test various kinds of RNN with TensorFlow
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn

timesteps = 5

num_hidden = 8
num_input = 3
num_classes = 1
num_batches = 2

# tf Graph input
X_batchmajor = tf.placeholder("float", [None, timesteps, num_input])
X_timemajor = tf.placeholder("float", [timesteps, None, num_input])
#Y = tf.placeholder("float", [None, num_classes])

tf.set_random_seed(0)

with tf.variable_scope('rnn', initializer=tf.initializers.ones()):
    lstm_cell = rnn.BasicLSTMCell(num_units=num_hidden,
                                forget_bias=0.0, # 1.0?
                                activation=tf.nn.tanh)

                                # cudnn and tf recurrent activations may be different
                                # forget bias is also there...

    '''According to CuDNN docs the final activation is tanh and gate activations (recurrent_activation in Keras terminology) are sigmoid.'''

    # set...
    # rnn/basic_lstm_cell/kernel:0
    # rnn/basic_lstm_cell/bias:0

    # defining initial state
    initial_state = lstm_cell.zero_state(num_batches, dtype=tf.float32)

    outputs2, states2 = tf.nn.dynamic_rnn(lstm_cell,
                                         X_batchmajor,
                                         dtype=tf.float32,
                                         initial_state=initial_state,
                                         time_major=False)


    outputs, states = tf.nn.dynamic_rnn(lstm_cell,
                                        X_timemajor,
                                        dtype=tf.float32,
                                        initial_state=initial_state,
                                        time_major=True)

    cudnn_cell = cudnn_rnn.CudnnLSTM(num_layers=1,
                                     num_units=num_input,
                                     direction=cudnn_rnn.CUDNN_RNN_UNIDIRECTION,
                                     input_mode=cudnn_rnn.CUDNN_INPUT_LINEAR_MODE,
                                     name="CudnnLSTM",
                                     dropout=0.0,
                                     seed=0.0,
                                     kernel_initializer=tf.initializers.ones(),
                                     bias_initializer=tf.initializers.zeros(),
                                     dtype=tf.float32)

    cudnn_outputs, cudnn_states = cudnn_cell(
        inputs=X_timemajor, # 3-D tensor [time_len, batch_size, input_size]
        training=True
    )

#x = tf.unstack(X, timesteps, 1)
#outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

tvars = tf.trainable_variables()
with tf.Session() as sess:
    sess.run(init)

    tvars_vals = sess.run(tvars)

    print('Before')
    for var, val in zip(tvars, tvars_vals):
        print(var.name, val)  # Prints the name of the variable alongside its value.

    # desired shape:
    # original shape (2, 5, 3)  [batch, time, depth]
    # time-major (5, 2, 3)      [time, batch, depth]
    constval = np.asarray(
        [   # all batches
            [   # batch 1
                [1., 2., 3.],       # timestep 1: 3(num_input) inputs
                [1., 2., 3.],       # timestep 2
                [1., 2., 3.],       # timestep 3
                [1., 2., 3.],       # timestep 4
                [1., 2., 3.]        # timestep 5
            ], # ...
            [   # batch 2
                [1., 2., 3.],       # timestep 1: 3(num_input) inputs
                [1., 2., 3.],       # timestep 2
                [1., 2., 3.],       # timestep 3
                [1., 2., 3.],       # timestep 4
                [1., 2., 3.]        # timestep 5
            ], # ...
        ]
    )

    assert constval.shape[0] == num_batches
    assert constval.shape[1] == timesteps
    assert constval.shape[2] == num_input

    print('original', constval)
    print('original shape', constval.shape)

    #print(sess.run(m, feed_dict={x: [[2.]]}))
    #print(sess.run(outputs, feed_dict={X: constval}))

    constval_timemajor = np.transpose(constval, [1, 0, 2])
    print('time-major', constval_timemajor.shape)

    assert constval_timemajor.shape[0] == timesteps
    assert constval_timemajor.shape[1] == num_batches
    assert constval_timemajor.shape[2] == num_input

    # time-major
    print('TF outputs2', sess.run(outputs2, feed_dict={X_batchmajor: constval}))
    print('TF states2', sess.run(states2, feed_dict={X_batchmajor: constval}))
    print('TF outputs', sess.run(outputs, feed_dict={X_timemajor: constval_timemajor}))
    print('TF states', sess.run(states, feed_dict={X_timemajor: constval_timemajor}))
    print('CUDNN outputs', sess.run(cudnn_outputs, feed_dict={X_timemajor: constval_timemajor}))
    print('CUDNN states', sess.run(cudnn_states, feed_dict={X_timemajor: constval_timemajor}))

    print('After')
    for var, val in zip(tvars, tvars_vals):
        print(var.name, val)  # Prints the name of the variable alongside its value.

    #print(sess.run(outputs, feed_dict={X: [[[1.], [2.], [3.], [4.], [5.]]]}))
    #print(sess.run(outputs, feed_dict={X: [[[1., 2., 3., 4., 5.]]]}))
    #print(sess.run(outputs, feed_dict={X: [[[1.]], [[2.]], [[3.]], [[4.]], [[5.]]]}))
