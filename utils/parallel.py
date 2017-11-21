# parallel.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import six
import tensorflow as tf


def _maybe_repeat(x, n):
    if isinstance(x, list):
        assert len(x) == n
        return x
    else:
        return [x] * n


# data-level parallelism
def data_parallelism(devices, fn, *args, **kwargs):
    num_worker = len(devices)

    # replicate args and kwargs
    if args:
        new_args = [_maybe_repeat(arg, num_worker) for arg in args]
        # transpose
        new_args = [list(x) for x in zip(*new_args)]
    else:
        new_args = [[] for _ in xrange(num_worker)]

    new_kwargs = [{} for _ in xrange(num_worker)]

    for k, v in six.iteritems(kwargs):
        vals = _maybe_repeat(v, num_worker)

        for i in xrange(num_worker):
            new_kwargs[i][k] = vals[i]

    fns = _maybe_repeat(fn, num_worker)

    # Now make the parallel call.
    outputs = []

    for i in xrange(num_worker):
        with tf.name_scope('parallel_%d' % i):
            with tf.variable_scope(tf.get_variable_scope(),
                                   reuse=True if i > 0 else None,
                                   caching_device="/cpu:0"):
                with tf.device(devices[i]):
                    outputs.append(fns[i](*new_args[i], **new_kwargs[i]))

    if isinstance(outputs[0], tuple):
        outputs = list(zip(*outputs))
        outputs = tuple([list(o) for o in outputs])

    return outputs


def shard_features(features, device_list):
    num_datashards = len(device_list)

    sharded_features = dict()
    for k, v in six.iteritems(features):
        v = tf.convert_to_tensor(v)
        if not v.shape.as_list():
            v = tf.expand_dims(v, axis=-1)
            v = tf.tile(v, [num_datashards])
        with tf.device(v.device):
            sharded_features[k] = tf.split(v, num_datashards, 0)

    datashard_to_features = []

    for d in xrange(num_datashards):
        feat = {
            k: v[d] for k, v in six.iteritems(sharded_features)
        }
        datashard_to_features.append(feat)

    return datashard_to_features


def parallel_model(model_fn, features, params, mode, use_cpu=False):
    devices = ["gpu:%d" % d for d in params.device_list]

    if use_cpu:
        devices += ["cpu:0"]

    if len(devices) == 1:
        return model_fn(features, params)

    features = shard_features(features, devices)

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        loss = data_parallelism(devices, model_fn, features, params)
        return tf.add_n(loss) / len(loss)
    elif mode == tf.contrib.learn.ModeKeys.EVAL:
        loss, logits = data_parallelism(devices, model_fn, features, params)
        return tf.add_n(loss) / len(loss), tf.concat(logits, 0)
    elif mode == tf.contrib.learn.ModeKeys.INFER:
        predicts = data_parallelism(devices, model_fn, features, params)
        return tf.concat(predicts, 0)
    else:
        raise ValueError("Unknown mode")
