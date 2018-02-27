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
num_rows = 4
assert num_input % num_rows == 0, 'input must be divisible by row num'

num_classes = 1
num_batches = 2

# tf Graph input
X_batchmajor = tf.placeholder("float", [None, timesteps, num_input])
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

'''
# filters=7(all ones)
# this would be the filter that would get trained
f = tf.Variable(tf.constant(
    [   # all batches
        [    #FirstRow                                   #LastRow
            [1., 1., 1., 1., 1., 1., 1.],       # channel 1
            [1., 1., 1., 1., 1., 1., 1.],       # channel 2
            [1., 1., 1., 1., 1., 1., 1.],       # channel 3
            [1., 1., 1., 1., 1., 1., 1.],       # channel 4
            [1., 1., 1., 1., 1., 1., 1.],       # channel 5
            [1., 1., 1., 1., 1., 1., 1.],       # channel 6
            [1., 1., 1., 1., 1., 1., 1.],       # channel 7
            [1., 1., 1., 1., 1., 1., 1.],       # channel 8
            [1., 1., 1., 1., 1., 1., 1.],       # channel 9
            [1., 1., 1., 1., 1., 1., 1.],       # channel 10
            [1., 1., 1., 1., 1., 1., 1.],       # channel 11
            [1., 1., 1., 1., 1., 1., 1.]        # channel 12
        ]
    ]
), dtype=tf.float32, name="7filters_all_ones")

# filters=12
# this would be the filter that would get trained
f12 = tf.Variable(tf.constant(
    [   # all batches
        [    #FirstRow                                   #LastRow
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],       # channel 1
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],       # channel 2
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],       # channel 3
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],       # channel 4
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],       # channel 5
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],       # channel 6
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],       # channel 7
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],       # channel 8
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],       # channel 9
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],       # channel 10
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],       # channel 11
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]        # channel 12
        ]
    ]
), dtype=tf.float32, name="12filters_all_ones")

# filters=1
f1 = tf.Variable(tf.constant(
    [   # all batches
        [   #Row0
            [1.],       # channel 1
            [1.],       # channel 2
            [1.],       # channel 3
            [1.],       # channel 4
            [1.],       # channel 5
            [1.],       # channel 6
            [1.],       # channel 7
            [1.],       # channel 8
            [1.],       # channel 9
            [1.],       # channel 10
            [1.],       # channel 11
            [1.]        # channel 12
        ]
    ]
), dtype=tf.float32, name="1filter_all_ones")
'''

print('X_batchmajor', X_batchmajor.shape)
#NHWC
print('...batch:', X_batchmajor.shape[0])
print('...in_width:', X_batchmajor.shape[1])
print('...in_channels:', X_batchmajor.shape[2])

print('filter', f7t.shape)
#NHWC
print('...filter_width:', f7t.shape[0])
print('...in_channels:', f7t.shape[1])
print('...out_channels:', f7t.shape[2])

#from tensorflow.python.keras.initializers import Initializer
from tensorflow.python.ops.init_ops import Initializer, _assert_float_dtype
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import dtypes
class CustomKernelInitializer(Initializer):
    def __init__(self, dtype=dtypes.float32):
        self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        #normal = random_ops.random_normal(shape, self.mean, self.stddev,
        #    dtype, seed=self.seed)
        # do what you want with normal here
        #return normal
        return f7t

    def get_config(self):
        return {"dtype": self.dtype.name}

raw_cnn_1d_out = tf.nn.conv1d(X_batchmajor,
                              filters=f7t,
                              stride=1,
                              padding='VALID',
                              use_cudnn_on_gpu=True,
                              data_format='NHWC') # num_samples, height/width, channels
                              #data_format='NCHW') # num_samples, channels, height/width

# CNN is performing addition over time series
with tf.variable_scope('cnn', initializer=tf.initializers.zeros()):
    cnn_1d_inst = tf.layers.Conv1D(filters=7,    # seems to just expand result
                                  kernel_size=1, # filter size
                                  strides=1,     # ?? reduces size of result?
                                  activation=None,
                                  padding='valid',
                                  use_bias=True,
                                  name='conv1d',
                                  #kernel_initializer=CustomKernelInitializer,
                                  kernel_initializer=tf.constant_initializer(f7t_raw),
                                  data_format='channels_last') # batch, length, channels

'''
    cnn_1d_out = tf.layers.conv1d(X_batchmajor,
                                  filters=7,    # seems to just expand result
                                  kernel_size=1, # filter size
                                  strides=1,     # ?? reduces size of result?
                                  activation=None,
                                  padding='valid',
                                  use_bias=True,
                                  name='conv1d',
                                  data_format='channels_last') # batch, length, channels
'''

cnn_1d_out = cnn_1d_inst.apply(X_batchmajor)

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
    #print('TF outputs2', sess.run(outputs2, feed_dict={X_batchmajor: constval}))
    #print('TF states2', sess.run(states2, feed_dict={X_batchmajor: constval}))
    #print('TF outputs', sess.run(outputs, feed_dict={X_timemajor: constval_timemajor}))
    #print('TF states', sess.run(states, feed_dict={X_timemajor: constval_timemajor}))
    #print('CUDNN outputs', sess.run(cudnn_outputs, feed_dict={X_timemajor: constval_timemajor}))
    #print('CUDNN states', sess.run(cudnn_states, feed_dict={X_timemajor: constval_timemajor}))

    print('CNN outputs', sess.run(cnn_1d_out, feed_dict={X_batchmajor: constval}))
    print('Raw CNN outputs', sess.run(raw_cnn_1d_out, feed_dict={X_batchmajor: constval}))

    #print('After')
    #for var, val in zip(tvars, tvars_vals):
    #    print(var.name, val)  # Prints the name of the variable alongside its value.

    #print(sess.run(outputs, feed_dict={X: [[[1.], [2.], [3.], [4.], [5.]]]}))
    #print(sess.run(outputs, feed_dict={X: [[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 4., 5.]]]}))
    #print(sess.run(outputs, feed_dict={X: [[[1.]], [[2.]], [[3.]], [[4.]], [[5.]]]}))
