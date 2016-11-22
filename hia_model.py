import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import sparse_ops
from my_util import hia_inference

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('vocab_size', 129886, 'Vocabulary size')
flags.DEFINE_integer('embedding_size', 500, 'Embedding dimension')
flags.DEFINE_integer('hidden_size', 512, 'Hidden units')
flags.DEFINE_integer('batch_size', 38, 'Batch size')
flags.DEFINE_integer('epochs', 8, 'Number of epochs to train/test')
flags.DEFINE_boolean('training', False, 'Training or testing a model')
flags.DEFINE_string('name', '', 'Model name (used for statistics and model path')
flags.DEFINE_float('dropout_keep_prob', 0.5, 'Keep prob for embedding dropout')
flags.DEFINE_float('l2_reg', 0.0002, 'l2 regularization for embeddings')
# We used stochastic gradient descent with the ADAM update rule and learning rate of 0.001 or 0.0003, 
# with an initial learning rate of 0.001.

model_path = 'models/' + FLAGS.name

if not os.path.exists(model_path):
    os.makedirs(model_path)

def train(y_hat, regularizer, document, doc_weight, answer):
  # Trick while we wait for tf.gather_nd - https://github.com/tensorflow/tensorflow/issues/206
  # This unfortunately causes us to expand a sparse tensor into the full vocabulary
  index = tf.range(0, FLAGS.batch_size) * FLAGS.vocab_size + tf.to_int32(answer)
  flat = tf.reshape(y_hat, [-1])
  relevant = tf.gather(flat, index)
 
  # mean cause reg is independent of batch size
  loss = -tf.reduce_mean(tf.log(relevant)) + FLAGS.l2_reg * regularizer 

  global_step = tf.Variable(0, name="global_step", trainable=False)

  accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y_hat, 1), answer)))

  optimizer = tf.train.AdamOptimizer()
  grads_and_vars = optimizer.compute_gradients(loss)
  capped_grads_and_vars = [(tf.clip_by_value(grad, -5, 5), var) for (grad, var) in grads_and_vars]
  train_op = optimizer.apply_gradients(capped_grads_and_vars, global_step=global_step)

  tf.scalar_summary('loss', loss)
  tf.scalar_summary('accuracy', accuracy)
  return loss, train_op, global_step, accuracy

def read_records(index=0):
  train_queue = tf.train.string_input_producer(['training.tfrecords'], num_epochs=FLAGS.epochs)
  validation_queue = tf.train.string_input_producer(['validation.tfrecords'], num_epochs=FLAGS.epochs)
  test_queue = tf.train.string_input_producer(['test.tfrecords'], num_epochs=FLAGS.epochs)

  queue = tf.QueueBase.from_list(index, [train_queue, validation_queue, test_queue])
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
        'document': tf.VarLenFeature(tf.int64),
        'query': tf.VarLenFeature(tf.int64),
        'answer': tf.FixedLenFeature([], tf.int64)
      })

  document = sparse_ops.serialize_sparse(features['document'])
  query = sparse_ops.serialize_sparse(features['query'])
  answer = features['answer']

  document_batch_serialized, query_batch_serialized, answer_batch = tf.train.shuffle_batch(
      [document, query, answer], batch_size=FLAGS.batch_size,
      capacity=2000,
      min_after_dequeue=1000)

  sparse_document_batch = sparse_ops.deserialize_many_sparse(document_batch_serialized, dtype=tf.int64)
  sparse_query_batch = sparse_ops.deserialize_many_sparse(query_batch_serialized, dtype=tf.int64)

  document_batch = tf.sparse_tensor_to_dense(sparse_document_batch)
  document_weights = tf.sparse_to_dense(sparse_document_batch.indices, sparse_document_batch.shape, 1)

  query_batch = tf.sparse_tensor_to_dense(sparse_query_batch)
  query_weights = tf.sparse_to_dense(sparse_query_batch.indices, sparse_query_batch.shape, 1)

  return document_batch, document_weights, query_batch, query_weights, answer_batch

def main():
  dataset = tf.placeholder_with_default(0, [])
  document_batch, document_weights, query_batch, query_weights, answer_batch = read_records(dataset)

  y_hat, reg = hia_inference(document_batch, document_weights, query_batch, query_weights)
  loss, train_op, global_step, accuracy = train(y_hat, reg, document_batch, document_weights, answer_batch)
  summary_op = tf.merge_all_summaries()

  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter(model_path, sess.graph)
    saver_variables = tf.all_variables()
    if not FLAGS.training:
      saver_variables = filter(lambda var: var.name != 'input_producer/limit_epochs/epochs:0', saver_variables)
      saver_variables = filter(lambda var: var.name != 'smooth_acc:0', saver_variables)
      saver_variables = filter(lambda var: var.name != 'avg_acc:0', saver_variables)
    saver = tf.train.Saver(saver_variables)

    sess.run([
        tf.initialize_all_variables(),
        tf.initialize_local_variables()])
    model = tf.train.latest_checkpoint(model_path)
    if model:
      print('Restoring ' + model)
      saver.restore(sess, model)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    start_time = time.time()
    accumulated_accuracy = 0
    try:
      if FLAGS.training:
        while not coord.should_stop():
          loss_t, _, step, acc = sess.run([loss, train_op, global_step, accuracy], feed_dict={dataset: 0})
          elapsed_time, start_time = time.time() - start_time, time.time()
          print(step, loss_t, acc, elapsed_time)
          if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
          if step % 1000 == 0:
            saver.save(sess, model_path + '/aoa', global_step=step)
      else:
        step = 0
        while not coord.should_stop():
          acc = sess.run(accuracy, feed_dict={dataset: 2})
          step += 1
          accumulated_accuracy += (acc - accumulated_accuracy) / step
          elapsed_time, start_time = time.time() - start_time, time.time()
          print(accumulated_accuracy, acc, elapsed_time)
    except tf.errors.OutOfRangeError:
      print('Done!')
    finally:
      coord.request_stop()
    coord.join(threads)

    '''
    import pickle
    with open('counter.pickle', 'r') as f:
      counter = pickle.load(f)
    word, _ = zip(*counter.most_common())
    '''

if __name__ == "__main__":
  main()
