from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(1337)  # for reproducibility

import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

#USE_TPU = False
USE_TPU = True
#TPU_NAME = 'node-1'
TPU_NAME = 'ee01_short_v2'

MODEL_DIR = './MLP/model_1'
if USE_TPU:
    MODEL_DIR='gs://vinh-tutorial/output/MLP/model_1'

# Parameters
learning_rate = 0.01
num_steps = 5000
batch_size = 128
display_step = 100

# Network Parameters
num_input = 10000 
num_classes = 2 # recommendation outcome
HIDDEN_UNITS = [4096, 2048, 1024, 512, 256]
#HIDDEN_UNITS = [256, 256]

# Define the neural network
def neural_net(x, HIDDEN_UNITS):
    for num_units in HIDDEN_UNITS:
        x = tf.layers.dense(x, num_units)
    out_layer = tf.layers.dense(x, 1) #Binary output 
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


class MyInput(object): 
    def __init__(self, is_training=True, is_eval=True, N=10000, input_dim=1):
          self.is_training = is_training
          self.is_eval = is_eval
          inputs_1, outputs = self.createData(N, input_dim)
          self.outputs = np.asarray(outputs, 'float32')
          self.inputs_1 = np.asarray(inputs_1, 'float32')
    
    def createData(self, NUM_SAMPLES, input_dim):
        X = np.random.randn(NUM_SAMPLES, input_dim).astype(np.float32)
        
        #y = np.random.randint(0,num_classes,size=(NUM_SAMPLES))
        y = np.expand_dims(np.sum(X,axis=1)>0, axis=1)
    
        return X, y.astype(np.float32)

    def input_fn(self, params):      
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

  train_data = MyInput(is_training=True, is_eval=False, N=10000, input_dim=num_input)
  eval_data  = MyInput(is_training=False, is_eval=True, N=2560, input_dim=num_input)
  test_data  = MyInput(is_training=False, is_eval=False, N=2560, input_dim=num_input)  
  
  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=USE_TPU,
      config=run_config,
      train_batch_size=256,
      eval_batch_size=256,
      predict_batch_size=256,
      )
  
  estimator.train(input_fn=train_data.input_fn, max_steps=10000)


  #testing      
  preds = estimator.evaluate(input_fn=eval_data.input_fn, steps=10, 
                            #yield_single_examples=False
                            )
  
  print(preds)
  
  preds = estimator.predict(input_fn=test_data.input_fn,
                            #yield_single_examples=False
                            )
    
if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
