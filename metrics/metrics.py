# metrics.py
import tensorflow as tf


# predictions: [batch, max_length]
# labels: [batch, max_length]
def calculate_precision(predictions, labels):
    mask = tf.cast(tf.not_equal(predictions, 0), tf.float32)
    matched = tf.cast(tf.equal(predictions, labels), tf.float32)

    return tf.reduce_sum(matched * mask) / tf.reduce_sum(mask)


def calculate_recall(predictions, labels):
    mask = tf.cast(tf.not_equal(predictions, 0), tf.float32)
    matched = tf.cast(tf.equal(predictions, labels), tf.float32)
    non_zero = tf.cast(tf.not_equal(labels, 0), tf.float32)

    return tf.reduce_sum(matched * mask) / tf.reduce_sum(non_zero)


def calculate_fmeasure(predictions, labels):
    precision = calculate_precision(predictions, labels)
    recall = calculate_recall(predictions, labels)

    return 2 * precision * recall / (precision + recall)


def precision_metric_fn(predictions, labels, weights=None):
    values = calculate_precision(predictions, labels)
    weights = tf.constant(1.0)

    return tf.metrics.mean(values, weights)


def recall_metric_fn(predictions, labels, weights=None):
    values = calculate_recall(predictions, labels)
    weights = tf.constant(1.0)

    return tf.metrics.mean(values, weights)


def f_metric_fn(predictions, labels, weights=None):
    values = calculate_fmeasure(predictions, labels)
    weights = tf.constant(1.0)

    return tf.metrics.mean(values, weights)


def create_tagger_evaluation_metrics():
    f_spec = tf.contrib.learn.MetricSpec(f_metric_fn,
                                         prediction_key="predictions")
    p_spec = tf.contrib.learn.MetricSpec(precision_metric_fn,
                                         prediction_key="predictions")
    r_spec = tf.contrib.learn.MetricSpec(recall_metric_fn,
                                         prediction_key="predictions")

    return {"precision": p_spec, "recall": r_spec, "f-measure": f_spec}
