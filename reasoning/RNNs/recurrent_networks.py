import tensorflow as tf
import numpy as np

# Elman dynamics
class RNN():
    def __init__(self, input_size, hidden_size, add_noise=False):
        xav_init         = tf.contrib.layers.xavier_initializer
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.W           = tf.get_variable('W', shape=[hidden_size, hidden_size], initializer=xav_init())
        self.U           = tf.get_variable('U', shape=[input_size, hidden_size], initializer=xav_init())
        self.b           = tf.get_variable('b', shape=[hidden_size], initializer=tf.constant_initializer(0.))
        self.add_noise   = add_noise
    
    # step operation
    def step(self, tuple_, x):
        st_1, _ = tuple_
        if self.add_noise:
            x, noise_h = x[:,:self.input_size], x[:,-self.hidden_size:]        
        # update hidden state
        ht = tf.matmul(st_1, self.W) + tf.matmul(x, self.U) + self.b
        ht_exact    = ht
        if self.add_noise:            
            noise_added = noise_h
            noise_added = tf.stop_gradient(noise_added)
            ht          = ht + noise_added
        return tf.tanh(ht),  tf.reduce_mean(tf.math.abs(ht - ht_exact))
