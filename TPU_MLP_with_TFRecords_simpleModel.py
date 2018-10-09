from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import os
np.random.seed(1337)  # for reproducibility

import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

#USE_TPU = False
USE_TPU = True
TPU_NAME = 'node-us'

MODEL_DIR = './MLP/model_simple'
DATA_DIR = './MLP_data'

if USE_TPU:
    MODEL_DIR='gs://vinh-tutorial/output/MLP/model_simple'
    DATA_DIR = 'gs://vinh-tutorial/data/MLP_data'

prefetch_buffer_size = 128
num_files_infeed = 16
shuffle_buffer_size = 512
num_parallel_calls = 32


# Parameters
learning_rate = 0.01
num_steps = 10000
batch_size = 256
display_step = 1000

# Network Parameters
num_input = 10000 
num_classes = 2 # recommendation outcome

#HIDDEN_UNITS = [4096, 2048, 1024, 512, 256]
HIDDEN_UNITS = [256, 256] #logistic regression

# Define the neural network
def neural_net(x, HIDDEN_UNITS):
    for num_units in HIDDEN_UNITS:
        x = tf.layers.dense(x, num_units, activation='relu')
    out_layer = tf.layers.dense(x, 1, activation=None) #Binary output 
    return out_layer

def model_fn(features, labels, mode, params):
  """Define a simple Dense attention model in Keras."""
  del params  # unused
  
  # Build the neural network
  logits = neural_net(features, HIDDEN_UNITS)

  if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_classes = tf.greater(tf.sigmoid(logits), 0.5)
        predicted_classes = tf.cast(predicted_classes, tf.int32)
        predictions = {
            'class_ids': predicted_classes,
            'probabilities': tf.nn.sigmoid(logits),
            'logits': logits,            
        }
        return tpu_estimator.TPUEstimatorSpec(mode, predictions=predictions)       

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

  
class MLP_Input(object):
  """Wrapper class that acts as the input_fn to TPUEstimator."""

  def __init__(self, is_training=True, is_eval=True, data_dir=None):
    self.is_eval = is_eval
    self.is_training = is_training
    self.data_dir = data_dir if data_dir else DATA_DIR

  def dataset_parser(self, value):
    """Parse an Imagenet record from value."""
    keys_to_features = {
        'X': tf.FixedLenFeature([], dtype=tf.string),
        'y': tf.FixedLenFeature(shape=[1], dtype=tf.int64)            
    }
    parsed = tf.parse_single_example(value, keys_to_features)
    X = tf.decode_raw(parsed['X'], tf.float32)
    X = tf.reshape(X, [10000])
    
    y = tf.cast(parsed['y'], tf.float32)
    return X, y

  def __call__(self, params):
    """Input function which provides a single batch for train or eval."""
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # `tf.contrib.tpu.RunConfig` for details.
    batch_size = params['batch_size']

    # Shuffle the filenames to ensure better randomization
    file_pattern = os.path.join(
        self.data_dir, 'MLP_data_train*' if self.is_training 
        else 'MLP_data_test*' )
    dataset = tf.data.Dataset.list_files(file_pattern)
    if self.is_training:
      dataset = dataset.shuffle(buffer_size=128)  # 1024 files in dataset

    if self.is_training or self.is_eval:
      dataset = dataset.repeat()

    def prefetch_dataset(filename):
      buffer_size =  prefetch_buffer_size
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            prefetch_dataset, cycle_length=num_files_infeed,
            sloppy=True))
    dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.map(
        self.dataset_parser,
        num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(32) #.cache()  # Prefetch overlaps in-feed with training
    #dataset = dataset.take(1).cache().repeat()

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
          iterations_per_loop=100,
          num_shards=8),
  )

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=USE_TPU,
      config=run_config,
      train_batch_size=batch_size,
      eval_batch_size=1000,
      predict_batch_size=1000,
      )
  

  current_step = 0
  train_input = MLP_Input(True)
  test_input = MLP_Input(False, True)
  
  while current_step < num_steps:
    # Train for up to steps_per_eval number of steps.
    # At the end of training, a checkpoint will be written to --model_dir.
    next_checkpoint = min(current_step + display_step,
                          num_steps)
    start_time = time.time()
    estimator.train(input_fn=train_input, max_steps=next_checkpoint)
    current_step = next_checkpoint
    e_time = time.time() - start_time
    print('>>>>>>>>Sample per sec: %.1f'%(display_step*batch_size/e_time))

    # Evaluate the model on the most recent model in --model_dir.
    # Since evaluation happens in batches of --eval_batch_size, some images
    # may be consistently excluded modulo the batch size.
    tf.logging.info('Starting to evaluate.')
    eval_results = estimator.evaluate(
                    input_fn=test_input,
                    steps=1)
    tf.logging.info('Eval results: %s' % eval_results)
    

  #testing    
  print('>>>>>>>>>>>>>Testing...')  
  preds = estimator.evaluate(input_fn=test_input, steps=10, 
                            #yield_single_examples=False
                            )
  
  print(preds)
  
  #preds = estimator.predict(input_fn=MLP_Input.input_fn,
                            #yield_single_examples=False
  #                          )
    
if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
