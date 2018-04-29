# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:33:05 2018

@author: xnguyen
"""

import tensorflow as tf
import itertools


def gen():
  for i in itertools.count(1):
    yield (i, [1] * i)

ds = tf.data.Dataset.from_generator(
    gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))
value = ds.make_one_shot_iterator().get_next()

with tf.Session() as sess:
  for i in range(10):    
    print(sess.run(value))  # (1, array([1]))
    #print(sess.run(value)) # (2, array([1, 1]))


# Using the generator pattern (an iterable)
class generator_recurrent_sin(object):
    def __init__(self, n, time_steps, input_dim, attention_column=13):
        self.n = n        
        self.time_steps = time_steps
        self.input_dim = input_dim        
        self.attention_column = attention_column
        self.num = 0
        
    def __iter__(self):
        return self.next()
            
    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n:
            self.num += 1
            x = np.random.standard_normal(size=(self.time_steps, self.input_dim))   
            y = np.random.randint(low=0, high=2, size=(1))
         
            freq = 0
            if y ==0:
                freq = 0.1 * PI
            else:
                freq = 0.5 * PI
            for t in range(self.attention_column, int(self.attention_column+self.input_dim/2)):
                for d in range(self.attention_column):
                    x[t, d] = np.sin(t*freq) + 0.05*np.random.randn(1)
            yield (x, y)
        else:            
            raise StopIteration()

INPUT_DIM = 32
TIME_STEPS = 64
HIDDEN_UNITS = 256
ATTENTION_COLUMN =10
N=1000

def get_data_generator():
      return generator_recurrent_sin(N, TIME_STEPS, INPUT_DIM,
                                             ATTENTION_COLUMN)
  
dataset = tf.data.Dataset.from_generator(get_data_generator,
                                           (tf.float32, tf.float32),
                                           (tf.TensorShape([TIME_STEPS, INPUT_DIM]),
                                            tf.TensorShape([1])))
batch_size=10
  
dataset = dataset.shuffle(buffer_size=1024)   # 1024 files in dataset
dataset = dataset.repeat()          
dataset = dataset.apply(
          tf.contrib.data.batch_and_drop_remainder(batch_size)
      )
          
dataset = dataset.prefetch(4)
images, labels = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    print(sess.run([images, labels]))
