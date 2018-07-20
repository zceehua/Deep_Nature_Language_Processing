import tensorflow as tf
import numpy as np
import re

_no_value=object()
def get_embedding(embedding, params):
    embedding_tf = tf.Variable(tf.zeros(shape=(params['vocab_size'], params['embed_dim']), dtype=tf.float32),
                               trainable=False,
                               name="embedding_matrix_result",
                               collections=[tf.GraphKeys.LOCAL_VARIABLES])

    update_op = tf.assign_add(embedding_tf, embedding)
    return tf.convert_to_tensor(embedding_tf), update_op

def model_fn(features, labels, mode, params):
    W = tf.get_variable('softmax_W', [params['vocab_size'], params['embed_dim']])
    b = tf.get_variable('softmax_b', [params['vocab_size']])
    if params["embedding"] is _no_value:
        E = tf.get_variable('embedding', [params['vocab_size'], params['embed_dim']])
    else:
        E=tf.Variable(initial_value=params["embedding"],
                                name="embedding",
                                dtype="float32")
    embedded = tf.nn.embedding_lookup(E, features["x"])  # forward activation

    loss_op = tf.reduce_mean(tf.nn.sampled_softmax_loss(
        weights=W,
        biases=b,
        labels=labels,
        inputs=embedded,
        num_sampled=params['n_sampled'],
        num_classes=params['vocab_size']))

    train_op = tf.train.AdamOptimizer().minimize(
        loss_op, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        embedding = get_embedding(E, params)
        metrics = {
            "embedding": embedding
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss_op, eval_metric_ops=metrics)

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #
    #     return tf.estimator.EstimatorSpec(mode, predictions=predictions)