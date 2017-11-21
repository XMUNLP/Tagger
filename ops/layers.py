# layers.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf

from common import check_data_format


def _linear_2d(inputs, output_size, bias, concat=True):
    input_size = [item.get_shape()[1].value for item in inputs]

    outputs = []

    if concat:
        input_size = sum(input_size)
        inputs = tf.concat(inputs, 1)

        shape = [input_size, output_size]
        matrix = tf.get_variable("matrix", shape)
        outputs.append(tf.matmul(inputs, matrix))
    else:
        for i in range(len(input_size)):
            shape = [input_size[i], output_size]
            name = "matrix_%d" % i
            matrix = tf.get_variable(name, shape)
            outputs.append(tf.matmul(inputs[i], matrix))

    output = tf.add_n(outputs)

    if bias is not None:
        shape = [output_size]
        bias = tf.get_variable("bias", shape)
        output = tf.nn.bias_add(output, bias)

    return output


def _linear_3d(inputs, output_size, bias, concat=True, data_format="NHWC"):
    data_format = check_data_format(data_format)
    channel_axis = 1 if data_format == "NCHW" else -1
    space_axis = -1 if data_format == "NCHW" else 1

    input_size = [item.get_shape()[channel_axis].value for item in inputs]

    outputs = []

    if concat:
        input_size = sum(input_size)
        inputs = tf.concat(inputs, channel_axis)
        inputs = tf.expand_dims(inputs, space_axis)

        shape = [input_size, output_size]
        matrix = tf.get_variable("matrix", shape)
        matrix = tf.expand_dims(tf.expand_dims(matrix, 0), 1)
        output = tf.nn.convolution(inputs, matrix, "VALID",
                                   data_format=data_format)
        outputs.append(output)
    else:
        for i in range(len(input_size)):
            inputs = tf.expand_dims(inputs, space_axis)

            shape = [input_size[i], output_size]
            name = "matrix_%d" % i
            matrix = tf.get_variable(name, shape)
            matrix = tf.expand_dims(tf.expand_dims(matrix, 0), 1)
            output = tf.nn.convolution(inputs, matrix, "VALID",
                                       data_format=data_format)
            outputs.append(output)

    output = tf.add_n(outputs)

    if bias is not None:
        bias = tf.get_variable("bias", [output_size])
        output = tf.nn.bias_add(output, bias, data_format=data_format)

    output = tf.squeeze(output, space_axis)

    return output


def _linear_4d(inputs, output_size, bias, concat=True, data_format="NHWC"):
    data_format = check_data_format(data_format)
    channel_axis = 1 if data_format == "NCHW" else -1

    input_size = [item.get_shape()[channel_axis].value for item in inputs]

    outputs = []

    if concat:
        input_size = sum(input_size)
        inputs = tf.concat(inputs, channel_axis)

        shape = [input_size, output_size]
        matrix = tf.get_variable("matrix", shape)
        matrix = tf.expand_dims(tf.expand_dims(matrix, 0), 1)
        output = tf.nn.convolution(inputs, matrix, "VALID",
                                   data_format=data_format)
        outputs.append(output)
    else:
        for i in range(len(input_size)):
            shape = [input_size[i], output_size]
            name = "matrix_%d" % i
            matrix = tf.get_variable(name, shape)
            matrix = tf.expand_dims(tf.expand_dims(matrix, 0), 1)
            output = tf.nn.convolution(inputs, matrix, "VALID",
                                       data_format=data_format)
            outputs.append(output)

    output = tf.add_n(outputs)

    if bias is not None:
        bias = tf.get_variable("bias", [output_size])
        output = tf.nn.bias_add(output, bias, data_format=data_format)

    return output


def _linear_5d(inputs, output_size, bias, concat=True, data_format="NHWC"):
    data_format = check_data_format(data_format)
    channel_axis = 1 if data_format == "NCHW" else -1

    input_size = [item.get_shape()[channel_axis].value for item in inputs]

    data_format = "NCDHW" if data_format is "NCHW" else "NDHWC"

    outputs = []

    if concat:
        input_size = sum(input_size)
        inputs = tf.concat(inputs, channel_axis)

        shape = [input_size, output_size]
        matrix = tf.get_variable("matrix", shape)
        matrix = tf.expand_dims(
                    tf.expand_dims(tf.expand_dims(matrix, 0), 1), 2
                )
        output = tf.nn.convolution(inputs, matrix, "VALID",
                                   data_format=data_format)
        outputs.append(output)
    else:
        for i in range(len(input_size)):
            shape = [input_size[i], output_size]
            name = "matrix_%d" % i
            matrix = tf.get_variable(name, shape)
            matrix = tf.expand_dims(
                tf.expand_dims(tf.expand_dims(matrix, 0), 1), 2
            )
            output = tf.nn.convolution(inputs, matrix, "VALID",
                                       data_format=data_format)
            outputs.append(output)

    output = tf.add_n(outputs)

    if bias is not None:
        bias = tf.get_variable("bias", [output_size])
        data_format = "NCHW" if data_format is "NCDHW" else "NHWC"
        output = tf.nn.bias_add(output, bias, data_format=data_format)

    return output


def linear(inputs, output_size, bias, concat=True, data_format="NHWC",
           dtype=None, scope=None):
    if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
    ndims = [ip.get_shape().ndims for ip in inputs]

    if any([dim - ndims[0] for dim in ndims]):
        raise ValueError("inputs do not agree on dimensions: %s" % ndims)

    rank = ndims[0]

    with tf.variable_scope(scope, default_name="linear", values=[inputs],
                           dtype=dtype):
        if rank == 2:
            output = _linear_2d(inputs, output_size, bias, concat)
        elif rank == 3:
            output = _linear_3d(inputs, output_size, bias, concat, data_format)
        elif rank == 4:
            output = _linear_4d(inputs, output_size, bias, concat, data_format)
        elif rank == 5:
            output = _linear_5d(inputs, output_size, bias, concat, data_format)
        else:
            raise ValueError("Input rank must be 2, 3 or 4, found %d" % rank)

        return output


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.name_scope(name, [x]):
        return tf.maximum(x, leak * x)


def batch_norm(*inputs, **kwargs):
    return tf.contrib.layers.batch_norm(*inputs, **kwargs)


def layer_norm(inputs, epsilon=1e-6, data_format="NHWC", dtype=None,
               scope=None):
    with tf.variable_scope(scope, default_name="layer_norm", values=[inputs],
                           dtype=dtype):
        data_format = check_data_format(data_format)
        axis = 1 if data_format == "NCHW" else -1
        channel_size = inputs.get_shape().as_list()[axis]

        scale = tf.get_variable("scale", shape=[channel_size],
                                initializer=tf.ones_initializer())

        offset = tf.get_variable("offset", shape=[channel_size],
                                 initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, axis=axis, keep_dims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=axis,
                                  keep_dims=True)

        norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)

        return norm_inputs * scale + offset


def smoothed_softmax_cross_entropy_with_logits(**kwargs):
    logits = kwargs.get("logits")
    labels = kwargs.get("labels")
    label_smoothing = kwargs.get("label_smoothing") or 0.0
    normalize = kwargs.get("normalize")

    if logits is None or labels is None:
        raise ValueError("Both logits and labels must be provided")

    if not label_smoothing:
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels
        )
        return ce

    # label smoothing
    vocab_size = tf.shape(logits)[1]
    labels = tf.reshape(labels, [-1])

    n = tf.to_float(vocab_size - 1)
    p = 1.0 - label_smoothing
    q = label_smoothing / n

    # Soft targets.
    soft_targets = tf.one_hot(tf.cast(labels, tf.int32), depth=vocab_size,
                              on_value=p, off_value=q)
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=soft_targets)

    if normalize is False:
        return xentropy

    # Normalizing constant is the best cross-entropy value with soft targets.
    # We subtract it just for readability, makes no difference on learning.
    normalizing = -(p * tf.log(p) + n * q * tf.log(q + 1e-20))

    return xentropy - normalizing


# 2D convolution
def conv2d(inputs, filter_height, filter_width, output_channel, padding,
           strides=None, dilation_rate=None, data_format="NCHW", dtype=None,
           scope=None):
    with tf.variable_scope(scope or "conv2d", dtype=dtype):
        if data_format == "NCHW":
            input_channel = inputs.get_shape().as_list()[1]
        else:
            input_channel = inputs.get_shape().as_list()[-1]

        filter_shape = [filter_height, filter_width, input_channel,
                        output_channel]
        filter_var = tf.get_variable("filter", filter_shape)

        if strides is None:
            strides = [1, 1]
        else:
            strides = list(strides)

        output = tf.nn.convolution(inputs, filter_var, padding, strides,
                                   dilation_rate, data_format)

        if data_format == "NCHW":
            output.set_shape([None, output_channel, None, None])
        else:
            output.set_shape([None, None, None, output_channel])

    return output


# transposed 2D convolution
def tconv2d(inputs, filter_height, filter_width, output_channel, padding,
            strides=None, data_format="NCHW", dtype=None, scope=None):
    with tf.variable_scope(scope or "tconv2d", dtype=dtype):
        if data_format == "NCHW":
            input_channel = inputs.get_shape().as_list()[1]
        else:
            input_channel = inputs.get_shape().as_list()[-1]

        filter_shape = [filter_height, filter_width, output_channel,
                        input_channel]

        filter_var = tf.get_variable("filter", filter_shape)

        if strides is None:
            strides = [1, 1, 1, 1]
        else:
            strides = list(strides)

        if len(strides) == 2:
            if data_format == "NCHW":
                strides = [1, 1] + strides
            else:
                strides = [1] + strides + [1]

        if data_format == "NCHW":
            sh, sw = strides[-2:]
        else:
            sh, sw = strides[1:-1]

        if padding == "SAME":
            batch = tf.shape(inputs)[0]
            if data_format == "NCHW":
                input_height = tf.shape(inputs)[2]
                input_width = tf.shape(inputs)[3]
                output_shape = [batch, output_channel, sh * input_height,
                                sw * input_width]
            else:
                input_height = tf.shape(inputs)[1]
                input_width = tf.shape(inputs)[2]
                output_shape = [batch, sh * input_height, sw * input_width,
                                output_channel]
        else:
            raise ValueError("Not implemented")

        output = tf.nn.conv2d_transpose(inputs, filter_var, output_shape,
                                        strides, padding, data_format)

        if data_format == "NCHW":
            output.set_shape([None, output_channel, None, None])
        else:
            output.set_shape([None, None, None, output_channel])

        return output
