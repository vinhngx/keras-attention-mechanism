#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import pandas as pd
import numpy as np
import pdb

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# In[2]:


def createData(NUM_SAMPLES, feature_columns, num_output=3):
    X = pd.DataFrame(np.random.randn(NUM_SAMPLES, len(feature_columns)), columns=feature_columns)
    y = np.random.randint(0,2,size=(NUM_SAMPLES))
    
    return X, y
    


# In[3]:


NUM_STEPS=100
BATCH_SIZE=256

NUM_FEATURES = 10000
NUM_TRAIN = 1000
NUM_TEST = 1000

HIDDEN_UNITS = [4096, 2048, 1024, 512, 256]

feature_columns = ['feature_%d'%i for i in range(NUM_FEATURES)]

# generate the data
(train_x, train_y) = createData(NUM_TRAIN, feature_columns, num_output=3)
(test_x, test_y)  = createData(NUM_TEST, feature_columns, num_output=3)


# In[ ]:


# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.DNNClassifier(
    feature_columns=[tf.feature_column.numeric_column(x) for x in feature_columns],
    # Two hidden layers of 10 nodes each.
    hidden_units=HIDDEN_UNITS,
    # The model must choose between 2 classes.
    n_classes=3)

# Train the Model.
classifier.train(
    input_fn=lambda:train_input_fn(train_x, train_y,
                                             BATCH_SIZE),
    steps=NUM_STEPS)

# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x, test_y,
                                            BATCH_SIZE))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


# In[ ]:





# In[ ]:




