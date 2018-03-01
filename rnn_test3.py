"""
Test various kinds of NN in TensorFlow with unambiguous, clear, fully
explained simple examples
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn

timesteps = 5

num_hidden = 8
num_input = 12
num_rows = 4 # width
num_cols = 3 # height
assert num_input % num_rows == 0, 'input must be divisible by row num'
assert num_input % num_cols == 0, 'input must be divisible by col num'
assert num_rows*num_cols == num_input, 'dimensions must match'

num_classes = 1
num_batches = 2

# tf Graph input
X_batchmajor = tf.placeholder("float", [None, timesteps, num_input])
X_batchmajor_2d = tf.placeholder("float", [None, timesteps, num_cols, num_rows])

assert num_rows % 2 == 0 # for 3d testing
X_batchmajor_3d = tf.placeholder("float", [None, timesteps, num_cols, num_rows/2, num_rows/2])
X_timemajor = tf.placeholder("float", [timesteps, None, num_input])
#Y = tf.placeholder("float", [None, num_classes])

tf.set_random_seed(0)

f7t_raw = [   # all batches
        [    #FirstRow                                   #LastRow
            [1., 1., 1., 1., 0., 1., 1.],       # channel 1
            [1., 1., 1., 1., 0., 1., 1.],       # channel 2
            [1., 1., 1., 1., 0., 1., 1.],       # channel 3
            [1., 1., 1., 1., 0., 1., 1.],       # channel 4
            [1., 1., 1., 1., 0., 1., 1.],       # channel 5
            [1., 1., 1., 0., 0., 1., 1.],       # channel 6
            [1., 1., 1., 0., 0., 1., 1.],       # channel 7
            [1., 1., 1., 1., 0., 1., 1.],       # channel 8
            [1., 1., 1., 1., 0., 1., 1.],       # channel 9
            [1., 1., 1., 1., 0., 1., 1.],       # channel 10
            [1., 1., 1., 1., 0., 1., 1.],       # channel 11
            [1., 1., 1., 1., 0., 1., 1.]        # channel 12
        ]
    ]

# filters=7(trained)
# this would be the filter that would get trained
f7t = tf.Variable(tf.constant(f7t_raw), dtype=tf.float32, name="7filters_trained")

f3x4t_raw = [   # all batches
        [    #FirstRow                                   #LastRow
            [
                [1., 1., 0., 1.],       # channel 1
                [1., 0., 1., 1.],       # channel 2
                [1., 1., 1., 1.],       # channel 3
                [1., 0., 1., 1.],       # channel 4
            ],
            [
                [1., 0., 1., 1.],       # channel 5
                [0., 1., 1., 0.],       # channel 6
                [0., 1., 1., 0.],       # channel 7
                [1., 1., 1., 1.],       # channel 8
            ],
            [
                [0., 1., 0., 1.],       # channel 9
                [1., 1., 1., 1.],       # channel 10
                [1., 0., 1., 1.],       # channel 11
                [1., 1., 1., 1.]        # channel 12
            ]
        ]
    ]

f3x4t = tf.Variable(tf.constant(f3x4t_raw), dtype=tf.float32, name="3x4filters_trained")

f3x2x2t_raw = [   # all batches
        [    #FirstRow                                   #LastRow
            [
                [
                    [1., 1., 0., 1.],       # channel 1
                    [1., 0., 1., 1.],       # channel 2
                ],
                [
                    [1., 1., 1., 1.],       # channel 3
                    [1., 0., 1., 1.],       # channel 4
                ]
            ],
            [
                [
                    [1., 0., 1., 1.],       # channel 5
                    [0., 1., 1., 0.],       # channel 6
                ],
                [
                    [0., 1., 1., 0.],       # channel 7
                    [1., 1., 1., 1.],       # channel 8
                ]
            ],
            [
                [
                    [0., 1., 0., 1.],       # channel 9
                    [1., 1., 1., 1.],       # channel 10
                ],
                [
                    [1., 0., 1., 1.],       # channel 11
                    [1., 1., 1., 1.]        # channel 12
                ]
            ]
        ]
    ]

'''
f3x2x2t_raw = [   # all batches
        [    #FirstRow                                   #LastRow
            [
                [
                    [1., 1.],       # channel 1  [ may have to delete inner portion a bit ]
                    [0., 1.],       # channel 2
                ],
                [
                    [1., 0.],       # channel 3
                    [1., 1.],       # channel 4
                ]
            ],
            [
                [
                    [1., 1.],       # channel 5  [ may have to delete inner portion a bit ]
                    [1., 1.],       # channel 6
                ],
                [
                    [1., 0.],       # channel 7
                    [1., 1.],       # channel 8
                ]
            ],
            [
                [
                    [1., 0.],       # channel 9  [ may have to delete inner portion a bit ]
                    [1., 1.],       # channel 10
                ],
                [
                    [0., 1.],       # channel 11
                    [1., 0.],       # channel 12
                ]
            ]
        ]
    ]
'''
f3x2x2t = tf.Variable(tf.constant(f3x2x2t_raw), dtype=tf.float32, name="3x2x2filters_trained")

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

print('X_batchmajor', X_batchmajor.shape)
#NHWC
print('...batch:', X_batchmajor.shape[0])
print('...in_width:', X_batchmajor.shape[1])
print('...in_channels:', X_batchmajor.shape[2])

print('1d filter', f7t.shape)
#NHWC
print('...filter_width:', f7t.shape[0])
print('...in_channels:', f7t.shape[1])
print('...out_channels:', f7t.shape[2])

print('2d filter', f3x4t.shape)
#NHWC
print('...filter_height:', f3x4t.shape[0])
print('...filter_width:', f3x4t.shape[1])
print('...in_channels:', f3x4t.shape[2])
print('...out_channels:', f3x4t.shape[3])

print('3d filter', f3x2x2t.shape)
#NDHWC
print('...filter_depth:', f3x2x2t.shape[0])
print('...filter_height:', f3x2x2t.shape[1])
print('...filter_width:', f3x2x2t.shape[2])
print('...in_channels:', f3x2x2t.shape[3])
print('...out_channels:', f3x2x2t.shape[4])

raw_cnn_1d_out = tf.nn.conv1d(X_batchmajor,
                              filters=f7t,
                              stride=1,
                              padding='VALID',
                              use_cudnn_on_gpu=True,
                              data_format='NHWC') # num_samples, height/width, channels
                              #data_format='NCHW') # num_samples, channels, height/width

raw_cnn_2d_out = tf.nn.conv2d(X_batchmajor_2d,
                              filter=f3x4t,
                              strides=(1, 1, 1, 1),
                              padding='VALID',
                              use_cudnn_on_gpu=True,
                              data_format='NHWC') # num_samples, height/width, channels

raw_cnn_3d_out = tf.nn.conv3d(X_batchmajor_3d,
                              filter=f3x2x2t,
                              strides=(1, 1, 1, 1, 1),
                              padding='VALID',
                              data_format='NDHWC') # num_samples, depth, height/width, channels


# CNN is performing addition over time series
with tf.variable_scope('cnn', initializer=tf.initializers.zeros()):
    cnn_1d_inst = tf.layers.Conv1D(filters=int(f7t.shape[2]),
                                  kernel_size=int(f7t.shape[0]), # conv length
                                  strides=1,     # ?? reduces size of result?
                                  activation=None,
                                  padding='valid',
                                  use_bias=True,
                                  name='conv1d',
                                  #kernel_initializer=CustomKernelInitializer,
                                  kernel_initializer=tf.constant_initializer(f7t_raw),
                                  data_format='channels_last') # batch, length, channels

    cnn_2d_inst = tf.layers.Conv2D(filters=int(f3x4t.shape[3]), # out_channels
                                  kernel_size=(int(f3x4t.shape[0]), int(f3x4t.shape[1])), # conv H, W
                                  strides=(1, 1),
                                  activation=None,
                                  padding='valid',
                                  use_bias=True,
                                  name='conv2d',
                                  kernel_initializer=tf.constant_initializer(f3x4t_raw),
                                  data_format='channels_last') # batch, height, width, channels

    cnn_3d_inst = tf.layers.Conv3D(filters=int(f3x2x2t.shape[4]), # out_channels
                                  kernel_size=(int(f3x2x2t.shape[0]), int(f3x2x2t.shape[1]), int(f3x2x2t.shape[2])), # conv D, H, W
                                  strides=(1, 1, 1),
                                  activation=None,
                                  padding='valid',
                                  use_bias=True,
                                  name='conv3d',
                                  kernel_initializer=tf.constant_initializer(f3x2x2t_raw),
                                  data_format='channels_last') # batch, height, width, channels

cnn_1d_out = cnn_1d_inst.apply(X_batchmajor)
cnn_2d_out = cnn_2d_inst.apply(X_batchmajor_2d)
cnn_3d_out = cnn_3d_inst.apply(X_batchmajor_3d)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

tvars = tf.trainable_variables()
with tf.Session() as sess:
    sess.run(init)

    tvars_vals = sess.run(tvars)

    print('Before')
    for var, val in zip(tvars, tvars_vals):
        print(var.name, val)  # Prints the name of the variable alongside its value.

    # in CNN, depth is called channels
    #         timesteps is called width/height (2D), or length (1D)
    
    # desired shape:
    # original shape (2, 5, 12)  [batch, time, depth]
    # time-major (5, 2, 12)      [time, batch, depth]
    constval = np.asarray(
        [   # all batches
            [   # batch 1
                [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],       # timestep 1: 12(num_input) inputs
                [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],       # timestep 2
                [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],       # timestep 3
                [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],       # timestep 4
                [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]        # timestep 5
            ], # ...
            [   # batch 2
                [1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1.],       # timestep 1: 12(num_input) inputs
                [1., 1., 1., 1., 1., 0.25, 1., 1., 1., 1., 1., 1.],       # timestep 2
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],       # timestep 3
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],       # timestep 4
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]        # timestep 5
            ], # ...
        ]
    )

    constval_3x4 = np.reshape(constval, [2, 5, 3, 4])
    constval_3x2x2 = np.reshape(constval, [2, 5, 3, 2, 2])

    assert constval.shape[0] == num_batches
    assert constval.shape[1] == timesteps
    assert constval.shape[2] == num_input

    print('original', constval)
    print('original shape', constval.shape)
    print('3x4', constval_3x4.shape)
    print('3x4 shape', constval_3x4.shape)
    print('3x2x2', constval_3x2x2.shape)
    print('3x2x2 shape', constval_3x2x2.shape)

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

    print('CNN 1D outputs', sess.run(cnn_1d_out, feed_dict={X_batchmajor: constval}))
    print('Raw CNN 1D outputs', sess.run(raw_cnn_1d_out, feed_dict={X_batchmajor: constval}))

    print('CNN 2D outputs', sess.run(cnn_2d_out, feed_dict={X_batchmajor_2d: constval_3x4}))
    print('Raw CNN 2D outputs', sess.run(raw_cnn_2d_out, feed_dict={X_batchmajor_2d: constval_3x4}))

    print('CNN 3D outputs', sess.run(cnn_3d_out, feed_dict={X_batchmajor_3d: constval_3x2x2}))
    print('Raw CNN 3D outputs', sess.run(raw_cnn_3d_out, feed_dict={X_batchmajor_3d: constval_3x2x2}))

    #print('After')
    #for var, val in zip(tvars, tvars_vals):
    #    print(var.name, val)  # Prints the name of the variable alongside its value.

    #print(sess.run(outputs, feed_dict={X: [[[1.], [2.], [3.], [4.], [5.]]]}))
    #print(sess.run(outputs, feed_dict={X: [[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 4., 5.]]]}))
    #print(sess.run(outputs, feed_dict={X: [[[1.]], [[2.]], [[3.]], [[4.]], [[5.]]]}))
