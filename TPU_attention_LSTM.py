from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(1337)  # for reproducibility

import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *



from attention_utils import get_data

INPUT_DIM = 2
TIME_STEPS = 20
HIDDEN_UNITS = 32
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False
LABEL_CLASSES = 2
USE_TPU = False
TPU_NAME = 'longjob-inceptionv4'
ATTENTION_COLUMN=13

MODEL_DIR = './LSTM_model_ckpt'
if USE_TPU:
    MODEL_DIR='gs://vinh-tutorial/output/RNN/LSTM_model_ckpt'
    

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    layers = tf.contrib.keras.layers
    
    input_dim = int(inputs.shape[2])
    a = layers.Permute((2, 1))(inputs)
    a = layers.Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = layers.Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = layers.Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = layers.RepeatVector(input_dim)(a)
    a_probs = layers.Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = layers.Multiply()([inputs, a_probs])
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

def model_fn(features, labels, mode, params):
  """Define a simple Dense attention model in Keras."""
  del params  # unused
  
  # Pass our input tensor to initialize the Keras input layer.
  layers = tf.contrib.keras.layers
  inputs = layers.Input(tensor=features)

  # ATTENTION PART STARTS HERE
  input_dim = int(inputs.shape[2])
  a = layers.Permute((2, 1))(inputs)
  a = layers.Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
  a = layers.Dense(TIME_STEPS, activation='softmax')(a)
  if SINGLE_ATTENTION_VECTOR:
        a = layers.Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = layers.RepeatVector(input_dim)(a)
  a_probs = layers.Permute((2, 1), name='attention_vec')(a)
  attention_mul = layers.Multiply()([inputs, a_probs])

  attention_mul = layers.LSTM(HIDDEN_UNITS, return_sequences=False)(attention_mul)
  
  logits = layers.Dense(1)(attention_mul)

  if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_classes = tf.greater(tf.sigmoid(logits), 0.5)
        predicted_classes = tf.cast(predicted_classes, tf.int32)
        predictions = {
            'class_ids': predicted_classes,
            'probabilities': tf.nn.sigmoid(logits),
            'logits': logits,
            'attention': a_probs,
        }
        return tpu_estimator.TPUEstimatorSpec(mode, predictions=predictions)       

  # Instead of constructing a Keras model for training, build our loss function
  # and optimizer in Tensorflow.
  #
  # N.B.  This construction omits some features that are important for more
  # complex models (e.g. regularization, batch-norm).  Once
  # `model_to_estimator` support is added for TPUs, it should be used instead.
  loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits, labels=labels
      )
  )
  optimizer = tf.train.AdamOptimizer()
  if USE_TPU:
    optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())


  if mode == tf.estimator.ModeKeys.TRAIN:
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          predictions={
              "classes": tf.argmax(input=logits, axis=1),
              "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
          }
      )
  elif mode == tf.estimator.ModeKeys.EVAL:        
      def metric_fn(labels, logits):
          """Evaluation metric function. Evaluates accuracy.
          Arguments should match the list of `Tensor` objects passed as the second
          element in the tuple passed to `eval_metrics`.
    
          Args:
            labels: `Tensor` with shape `[batch]`.
            logits: `Tensor` with shape `[batch, num_classes]`.
    
          Returns:
            A dict of the metrics to return from evaluation.
          """
          predicted_classes = tf.greater(tf.sigmoid(logits), 0.5)
          top_1_accuracy = tf.metrics.accuracy(labels, predicted_classes)
          
          return {
              'accuracy': top_1_accuracy,          
          }

      eval_metrics = (metric_fn, [labels, logits])
    
      return tpu_estimator.TPUEstimatorSpec(mode, 
                                        loss=loss,
                                        eval_metrics=eval_metrics
                                        )


class MyInput(object): 
    def __init__(self, is_training=True, N=100000):
          self.is_training = is_training
          inputs_1, outputs = get_data_recurrent(N, TIME_STEPS, INPUT_DIM,
                                                 ATTENTION_COLUMN)
          self.outputs = np.asarray(outputs, 'float32')
          self.inputs_1 = np.asarray(inputs_1, 'float32')
    
    def input_fn(self, params):
      """Read CIFAR input data from a TFRecord dataset."""        
      dataset = tf.data.Dataset.from_tensor_slices((self.inputs_1,
                                                    self.outputs))
      
      if self.is_training:
          dataset = dataset.shuffle(buffer_size=1024)   # 1024 files in dataset
          dataset = dataset.repeat()          
          
      dataset = dataset.apply(
          tf.contrib.data.batch_and_drop_remainder(256)
      )
          
      dataset = dataset.prefetch(4)
      images, labels = dataset.make_one_shot_iterator().get_next()
      return images, labels



def main(argv):

  tpu_grpc_url = None
  if USE_TPU:
        tpu_cluster_resolver = (
            tf.contrib.cluster_resolver.TPUClusterResolver(TPU_NAME))
        tpu_grpc_url = tpu_cluster_resolver.get_master()

  run_config = tpu_config.RunConfig(
      master=tpu_grpc_url,
      model_dir=MODEL_DIR,
      save_checkpoints_secs=3600,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=1000,
          num_shards=8),
  )

  train_data = MyInput(is_training=True, N=100000)
  test_data  = MyInput(is_training=False, N=2560)

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=USE_TPU,
      config=run_config,
      train_batch_size=256,
      eval_batch_size=256,
      predict_batch_size=256,
      )
  
  estimator.train(input_fn=train_data.input_fn, max_steps=20000)


  #testing      
  preds = estimator.evaluate(input_fn=test_data.input_fn, steps=10
                            #yield_single_examples=False
                            )
  
  print(preds)

  
  preds = estimator.predict(input_fn=test_data.input_fn, steps=10
                            #yield_single_examples=False
                            )
    
  sum_attn = []
  cnt = 0
  for i, pred in enumerate(preds):
      if i==0:
          sum_attn = pred['attention']
      else:
          sum_attn += pred['attention']
      cnt += 1
  print(sum_attn/cnt)
  
  sum_attn = np.mean(np.array(sum_attn), axis=1)
  print(sum_attn) 
  print(np.argmax(sum_attn), ATTENTION_COLUMN)
  
  """
  import matplotlib.pyplot as plt
  import pandas as pd
  pd.DataFrame(sum_attn, columns=['attention (%)']).plot(kind='bar', title='Attention Mechanism as '
                                                                           'a function of input'
                                                                           ' dimensions.')
  """


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
