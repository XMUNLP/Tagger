# attention.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import math
import tensorflow as tf

from layers import linear
from common import check_data_format


def _split_heads(x, num_heads, data_format="NHWC"):
    n = num_heads
    old_shape = x.get_shape().dims

    if data_format is "NCHW":
        x = tf.transpose(x, [0, 2, 1])

    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret, [0, 2, 1, 3])


def _combine_heads(x, data_format="NHWC"):
    x = tf.transpose(x, [0, 2, 1, 3])
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    x.set_shape(new_shape)

    if data_format is "NCHW":
        x = tf.transpose(x, [0, 2, 1])

    return x


def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a
    different frequency and phase.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(x+y) and cos(x+y) can
    be experessed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        x: a Tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float

    Returns:
        a Tensor the same shape as x.
    """
    with tf.name_scope("add_timing_signal", values=[x]):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )

        scaled_time = (tf.expand_dims(position, 1) *
                       tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])

        return x + signal


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """ Adds a bunch of sinusoids of different frequencies to a Tensor.

        Each channel of the input Tensor is incremented by a sinusoid of a
        different frequency and phase in one of the positional dimensions.

        This allows attention to learn to use absolute and relative positions.
        Timing signals should be added to some precursors of both the query and
        the memory inputs to attention.

        The use of relative position is possible because sin(a+b) and cos(a+b)
        can be experessed in terms of b, sin(a) and cos(a).

        x is a Tensor with n "positional" dimensions, e.g. one dimension for a
        sequence or two dimensions for an image

        We use a geometric sequence of timescales starting with min_timescale
        and ending with max_timescale.  The number of different timescales is
        equal to channels // (n * 2). For each timescale, we generate the two
        sinusoidal signals sin(timestep/timescale) and cos(timestep/timescale).
        All of these sinusoids are concatenated in the channels dimension.

        Args:
            x: a Tensor with shape [batch, d1 ... dn, channels]
            min_timescale: a float
            max_timescale: a float

        Returns:
            a Tensor the same shape as x.
    """
    static_shape = x.get_shape().as_list()
    num_dims = len(static_shape) - 2
    channels = tf.shape(x)[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1)
    )
    inv_timescales = min_timescale * tf.exp(
          tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
    )
    for dim in xrange(num_dims):
        length = tf.shape(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in xrange(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in xrange(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)
        x += signal

    return x


def add_positional_embedding_nd(x, max_length, name):
    """ Add n-dimensional positional embedding.

        Adds embeddings to represent the positional dimensions of the tensor.
        The input tensor has n positional dimensions - i.e. 1 for text, 2 for
        images, 3 for video, etc.

      Args:
          x: a Tensor with shape [batch, p1 ... pn, depth]
          max_length: an integer.  static maximum size of any dimension.
          name: a name for this layer.

      Returns:
          a Tensor the same shape as x.
    """
    static_shape = x.get_shape().as_list()
    dynamic_shape = tf.shape(x)
    num_dims = len(static_shape) - 2
    depth = static_shape[-1]
    base_shape = [1] * (num_dims + 1) + [depth]
    base_start = [0] * (num_dims + 2)
    base_size = [-1] + [1] * num_dims + [depth]
    for i in xrange(num_dims):
        shape = base_shape[:]
        start = base_start[:]
        size = base_size[:]
        shape[i + 1] = max_length
        size[i + 1] = dynamic_shape[i + 1]
        var = tf.get_variable(
            name + "_%d" % i,
            shape,
            initializer=tf.random_normal_initializer(0, depth ** -0.5)
        )
        var = var * (depth ** 0.5)
        x += tf.slice(var, start, size)
    return x


def attention_bias(inputs, mode, inf=-1e9, name="attention_bias"):
    with tf.name_scope(name, values=[inputs]):
        if mode == "incremental":
            length = inputs
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            ret = inf * (1.0 - lower_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        elif mode == "masking":
            mask = inputs
            ret = (1.0 - mask) * inf
            return tf.expand_dims(tf.expand_dims(ret, 1), 1)
        elif mode == "proximal":
            length = inputs
            r = tf.to_float(tf.range(length))
            diff = tf.expand_dims(r, 0) - tf.expand_dims(r, 1)
            m = tf.expand_dims(tf.expand_dims(-tf.log(1 + tf.abs(diff)), 0), 0)
            return m
        else:
            raise ValueError("Unknown mode %s" % mode)


def attention_image_summary(attn, image_shapes=None):
    """ Compute color image summary.

    Args:
        attn: a Tensor with shape
            [batch, num_heads, query_length, memory_length]
        image_shapes: optional tuple of integer scalars.
            If the query positions and memory positions represent the
            pixels of flattened images, then pass in their dimensions:
                (query_rows, query_cols, memory_rows, memory_cols).
            If the query positions and memory positions represent the
            pixels x channels of flattened images, then pass in their
            dimensions:
                (query_rows, query_cols, query_channels,
                memory_rows, memory_cols, memory_channels).
    """
    num_heads = attn.get_shape().as_list()[1]
    # [batch, query_length, memory_length, num_heads]
    image = tf.transpose(attn, [0, 2, 3, 1])
    image = tf.pow(image, 0.2)  # for high-dynamic-range
    # Each head will correspond to one of RGB.
    # pad the heads to be a multiple of 3
    image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, -num_heads % 3]])

    # split last dimensions
    n = 3
    old_shape = image.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    image = tf.reshape(image, tf.concat([tf.shape(image)[:-1], [n, -1]], 0))
    image.set_shape(new_shape)
    image = tf.reduce_max(image, 4)

    if image_shapes is not None:
        if len(image_shapes) == 4:
            q_rows, q_cols, m_rows, m_cols = list(image_shapes)
            image = tf.reshape(image, [-1, q_rows, q_cols, m_rows, m_cols, 3])
            image = tf.transpose(image, [0, 1, 3, 2, 4, 5])
            image = tf.reshape(image, [-1, q_rows * m_rows,
                                       q_cols * m_cols, 3])
        else:
            assert len(image_shapes) == 6
            q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels = list(
                    image_shapes
            )
            image = tf.reshape(image, [
                -1, q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels, 3
            ])
            image = tf.transpose(image, [0, 1, 4, 3, 2, 5, 6, 7])
            image = tf.reshape(image, [
                -1,
                q_rows * m_rows * q_channnels,
                q_cols * m_cols * m_channels,
                3
            ])
    tf.summary.image("attention", image, max_outputs=1)


def dot_product_attention(query, key, value, bias, keep_prob, summaries=False,
                          image_shapes=None, name=None):
    """ dot-product attention.

    Args:
        query: a Tensor with shape [batch, heads, length_q, depth_k]
        key: a Tensor with shape [batch, heads, length_kv, depth_k]
        value: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        keep_prob: a floating point number
        summaries: a boolean
        image_shapes: optional tuple of integer scalars.
            see comments for attention_image_summary()
        name: an optional string

    Returns:
        A Tensor.
    """
    with tf.name_scope(name, default_name="dot_product_attention",
                       values=[query, key, value]):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(query, key, transpose_b=True)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        if keep_prob is not None and keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)

        if summaries and not tf.get_variable_scope().reuse:
            attention_image_summary(weights, image_shapes)

        return tf.matmul(weights, value)


def additive_attention(query, key, value, bias, keep_prob, summaries=False,
                       image_shapes=None, name=None):
    """ dot-product attention.

    Args:
        query: a Tensor with shape [batch, heads, length_q, depth_k]
        key: a Tensor with shape [batch, heads, length_kv, depth_k]
        value: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        keep_prob: a floating point number
        summaries: a boolean
        image_shapes: optional tuple of integer scalars.
            see comments for attention_image_summary()
        name: an optional string

    Returns:
        A Tensor.
    """
    with tf.variable_scope(name, default_name="additive_attention",
                           values=[query, key, value]):
        query = tf.expand_dims(query, 3)
        key = tf.expand_dims(key, 2)

        hidden = query + key
        logits = linear(hidden, 1, False, scope="logits")
        logits = tf.squeeze(logits, -1)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        if keep_prob is not None and keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)

        if summaries and not tf.get_variable_scope().reuse:
            attention_image_summary(weights, image_shapes)

        return tf.matmul(weights, value)


def attention(query, key, value, bias, key_size, keep_prob=None,
              data_format="NHWC", summaries=False, image_shapes=None,
              dtype=None, scope=None):
    """ Standard attention.

    Args:
        query: a Tensor with shape [batch, length_q, depth_k] if
            data_format is 'NHWC' or shape [batch, depth_k, length_q] if
            data_format is 'NCHW' or [batch, length_q]
        key: a Tensor with shape [batch, length_kv, depth_k] if
            data_format is 'NHWC' or shape [batch, depth_k, length_kv]
            if data_format is 'NCHW'
        value: a Tensor with shape [batch, length_kv, depth_v] if
            data_format is 'NHWC' or shape [batch, depth_v, length_kv]
            if data_format is 'NCHW'
        key_size: hidden size
        bias: bias Tensor (see attention_bias())
        keep_prob: a floating point number
        data_format: data format used in convolution
        summaries: a boolean
        image_shapes: optional tuple of integer scalars.
            see comments for attention_image_summary()
        dtype: An instance of tf.DType
        scope: An optional string

    Returns:
        A Tensor.
    """

    with tf.variable_scope(scope, default_name="attention", dtype=dtype):
        if value is None:
            raise ValueError("value must not be None")

        if key is None:
            key = linear(value, key_size, True, data_format=data_format,
                         scope="key_transform")

            if query is None:
                return key

        # logits => query
        query = linear(query, key_size, True, data_format=data_format,
                       scope="query_transform")

        query_rank = query.get_shape().ndims

        if query_rank == 2:
            if data_format == "NCHW":
                query = tf.expand_dims(query, -1)
            else:
                query = tf.expand_dims(query, 1)

        # query: [batch, length_q, depth_k] or [batch, depth_k, length_q]
        # key: [batch, length_kv, depth_k] or [batch, depth_k, length_kv]
        if data_format == "NCHW":
            query = tf.expand_dims(query, -1)
            key = tf.expand_dims(key, 2)
        else:
            query = tf.expand_dims(query, 2)
            key = tf.expand_dims(key, 1)

        channel_axis = 1 if data_format == "NCHW" else -1
        hidden = tf.tanh(query + key)
        logits = linear(hidden, 1, False, scope="logits")
        # [batch, length_q, length_kv]
        logits = tf.squeeze(logits, channel_axis)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")

        # dropping out the attention links
        if keep_prob is not None and keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)

        if summaries and not tf.get_variable_scope().reuse:
            attention_image_summary(weights, image_shapes)

        # [batch, length_kv, depth_v] or [batch, depth_v, length_kv]
        if data_format == "NCHW":
            output = tf.matmul(weights, value, transpose_b=True)
        else:
            output = tf.matmul(weights, value)

        if query_rank == 2:
            output = tf.squeeze(output, 1)

        return output


def multihead_attention(query, memory, bias, key_size, value_size, output_size,
                        num_heads, keep_prob=None, data_format="NHWC",
                        attention_function="dot_product", summaries=False,
                        image_shapes=None, dtype=None, scope=None):
    """ Multihead scaled-dot-product attention with input/output
        transformations.

    Args:
        query: a Tensor with shape [batch, length_q, channels] if
            data_format is `NHWC`, [batch, channels, length_q] if
            data_format is `NCHW`
        memory: a Tensor with shape [batch, length_m, channels] if
            data_format is `NHWC`, [batch, channels, length_q] if
            data_format is `NCHW`
        bias: bias Tensor (see attention_bias())
        key_size: an integer
        value_size: an integer
        output_size: an integer
        num_heads: an integer dividing total_key_depth and total_value_depth
        keep_prob: a floating point number
        summaries: a boolean
        image_shapes: optional tuple of integer scalars.
            see comments for attention_image_summary()
        data_format: "NHWC" or "NCHW"
        attention_function: "dot_product" or "additive"
        dtype: an optional instance of tf.DType
        scope: an optional string

    Returns:
        A Tensor.
    """
    if key_size % num_heads != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_size, num_heads))

    if value_size % num_heads != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_size, num_heads))

    with tf.variable_scope(scope, default_name="multihead_attention",
                           values=[query, memory], dtype=dtype):
        data_format = check_data_format(data_format)
        axis = 1 if data_format is "NCHW" else 2

        if memory is None:
            # self attention
            size = key_size * 2 + value_size
            combined = linear(query, size, True, True, data_format=data_format,
                              scope="qkv_transform")
            q, k, v = tf.split(combined, [key_size, key_size, value_size],
                               axis=axis)
        else:
            q = linear(query, key_size, True, data_format=data_format,
                       scope="q_transform")
            combined = linear(memory, key_size + value_size, True,
                              data_format=data_format, scope="kv_transform")
            k, v = tf.split(combined, [key_size, value_size], axis=axis)

        # split heads
        q = _split_heads(q, num_heads, data_format=data_format)
        k = _split_heads(k, num_heads, data_format=data_format)
        v = _split_heads(v, num_heads, data_format=data_format)

        # scale query
        if attention_function == "dot_product":
            key_depth_per_head = key_size // num_heads
            q *= key_depth_per_head ** -0.5

            # attention
            x = dot_product_attention(q, k, v, bias, keep_prob, summaries,
                                      image_shapes)
        elif attention_function == "additive":
            x = additive_attention(q, k, v, bias, keep_prob, summaries,
                                   image_shapes)
        else:
            raise ValueError("Unknown attention function")

        # combine heads
        x = _combine_heads(x, data_format=data_format)

        x = linear(x, output_size, True, data_format=data_format,
                   scope="output_transform")
        return x
