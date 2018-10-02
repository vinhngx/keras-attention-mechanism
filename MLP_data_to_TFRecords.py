#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 11:41:18 2018

@author: xnguyen
"""

"""Convert dense Dataset to local TFRecords"""

import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import pdb

def _int64_feature(value:int) -> tf.train.Features.FeatureEntry:
    """Create a Int64List Feature
    
    Args:
        value: The value to store in the feature
    
    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value:str) -> tf.train.Features.FeatureEntry:
    """Create a BytesList Feature
    
    Args:
        value: The value to store in the feature
    
    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _data_path(data_directory:str, name:str) -> str:
    """Construct a full path to a TFRecord file to be stored in the 
    data_directory. Will also ensure the data directory exists
    
    Args:
        data_directory: The directory where the records will be stored
        name:           The name of the TFRecord
    
    Returns:
        The full path to the TFRecord file
    """
    if not os.path.isdir(data_directory):
        os.makedirs(data_directory)

    return os.path.join(data_directory, '%s.tfrecords'%name)


def convert_to(X, y, name:str, data_directory:str='./MLP_data', num_shards:int=128):
    """Convert the dataset into TFRecords on disk
    
    Args:
        X, y: dense data & label
        data_directory: The directory where records will be stored
        num_shards:     The number of files on disk to separate records into
    """
    num_examples = len(X)
    def _process_examples(start_idx:int, end_index:int, filename:str):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index in range(start_idx, end_index):
                sys.stdout.write("\rProcessing sample %s of %s"%(index+1,num_examples))
                sys.stdout.flush()

                image_raw = X[index].tostring()
                #pdb.set_trace()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'y': _int64_feature(int(y[index])),
                    'X': _bytes_feature(image_raw)
                }))
                writer.write(example.SerializeToString())
    
    if num_shards == 1:
        _process_examples(0, len(X), _data_path(data_directory, name))
    else:
        total_examples = len(X)
        samples_per_shard = total_examples // num_shards

        for shard in range(num_shards):
            start_index = shard * samples_per_shard
            end_index = start_index + samples_per_shard
            _process_examples(start_index, end_index, _data_path(data_directory, '%s-%s'%(name,shard+1)))

    print()
    
    
def createData(NUM_SAMPLES, feature_columns, num_classes):
    X = np.random.randn(NUM_SAMPLES, len(feature_columns)).astype(np.float32)
    
    #y = np.random.randint(0,num_classes,size=(NUM_SAMPLES))
    y = np.sum(X,axis=1)>0
    
    return X, y.astype(np.int64)
    
NUM_FEATURES = 10000
NUM_TRAIN = 1000
NUM_TEST = 1000
num_classes = 1

feature_columns = ['feature_%d'%i for i in range(NUM_FEATURES)]

# generate the data
for i in range(128):
    print(i)        
    (train_x, train_y) = createData(NUM_TRAIN, feature_columns, num_classes=num_classes)
    convert_to(train_x, train_y, 'MLP_data_train-%d'%i, data_directory='./MLP_data', num_shards=1)

(test_x, test_y)  = createData(NUM_TEST, feature_columns, num_classes=num_classes)
convert_to(test_x, test_y, 'MLP_data_test', data_directory='./MLP_data', num_shards=1)
