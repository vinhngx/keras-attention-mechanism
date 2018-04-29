from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(1337)  # for reproducibility

import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

from keras.layers.core import *
from keras.models import *

from attention_utils import get_data_recurrent, generator_recurrent_sin

INPUT_DIM = 256
TIME_STEPS = 64
HIDDEN_UNITS = 512
NUM_LAYERS = 3

# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
LABEL_CLASSES = 2
USE_TPU = False
TPU_NAME = 'longjob-inceptionv4'
ATTENTION_COLUMN=10

MODEL_DIR = './multi_LSTM_model_ckpt'
if USE_TPU:
    MODEL_DIR='gs://vinh-tutorial/output/RNN/multi_LSTM_model_ckpt'

def model_fn(features, labels, mode, params):
  """Define a simple Dense attention model in Keras."""
  del params  # unused
  
  # Pass our input tensor to initialize the Keras input layer.
  layers = tf.contrib.keras.layers
  inputs = layers.Input(tensor=features)
  input_sequence = tf.unstack(inputs, axis=1)
  
  #lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
  def lstm_cell():
      return tf.nn.rnn_cell.GRUCell(HIDDEN_UNITS)
      #return tf.nn.rnn_cell.LSTMCell(HIDDEN_UNITS, state_is_tuple=False)
      #return tf.contrib.rnn.BasicLSTMCell(HIDDEN_UNITS, state_is_tuple=False)
  cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(NUM_LAYERS)])
  
  #cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS, state_is_tuple=False)
   
  # Create an unrolled Recurrent Neural Networks to length of
  # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
  output_sequence, state = tf.nn.static_rnn(cell, input_sequence, dtype=tf.float32)
    
  # ATTENTION PART STARTS HERE
  output_sequence = tf.stack(output_sequence, axis=1)
  input_dim = int(output_sequence.shape[2])
  a = layers.Permute((2, 1))(output_sequence)
  a = layers.Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
  a = layers.Dense(TIME_STEPS, activation='softmax')(a)
  if SINGLE_ATTENTION_VECTOR:
        a = layers.Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = layers.RepeatVector(input_dim)(a)
  a_probs = layers.Permute((2, 1), name='attention_vec')(a)
  attention_mul = layers.Multiply()([output_sequence, a_probs])
  attention_mul = layers.Flatten()(attention_mul)
  
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
    def __init__(self, is_training=True, is_eval=True, N=10000, batch_size=256):
          self.is_training = is_training
          self.is_eval = is_eval                    
          self.batch_size = batch_size
          self.N = N
          inputs_1, outputs = get_data_recurrent(self.N, TIME_STEPS, INPUT_DIM, ATTENTION_COLUMN)
          self.inputs_1, self.outputs = np.asarray(inputs_1, 'float32'), np.asarray(outputs, 'float32')
    def input_fn(self, params):  
      batch_size = params['batch_size']
      
      
      dataset = tf.data.Dataset.from_tensor_slices((self.inputs_1, self.outputs))
      """
      def get_data_generator():
          return generator_recurrent_sin(self.N, TIME_STEPS, INPUT_DIM,
                                                 ATTENTION_COLUMN)
      
      dataset = tf.data.Dataset.from_generator(get_data_generator,
                                               (tf.float32, tf.float32),
                                               (tf.TensorShape([TIME_STEPS, INPUT_DIM]),
                                                tf.TensorShape([1])))
      """
      
      if self.is_training:
          dataset = dataset.shuffle(buffer_size=1024)   # 1024 files in dataset
          dataset = dataset.repeat()          
          
      dataset = dataset.apply(
          tf.contrib.data.batch_and_drop_remainder(batch_size)
      )
              
      dataset = dataset.prefetch(4)
      images, labels = dataset.make_one_shot_iterator().get_next()
      if self.is_training or self.is_eval:
          return images, labels
      else:
          return dataset



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

  train_data = MyInput(is_training=True, is_eval=False, N=1024)
  eval_data  = MyInput(is_training=False, is_eval=True, N=256)
  test_data  = MyInput(is_training=False, is_eval=False, N=256)  

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=USE_TPU,
      config=run_config,
      train_batch_size=1024,
      eval_batch_size=256,
      predict_batch_size=256,
      )
  
  estimator.train(input_fn=train_data.input_fn, max_steps=2000)


  #testing      
  acc = estimator.evaluate(input_fn=test_data.input_fn, steps=1
                            #yield_single_examples=False
                            )
  
  print(acc)

  
  preds = estimator.predict(input_fn=test_data.input_fn,
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
  print(acc)
  
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
