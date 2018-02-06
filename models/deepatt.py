# deepatt.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import ops
import copy
import tensorflow as tf

from ops.layers import layer_norm, linear
from ops.attention import multihead_attention


def deepatt_default_params():
    params = tf.contrib.training.HParams(
        feature_size=128,
        hidden_size=256,
        filter_size=1024,
        filter_width=3,
        num_heads=8,
        num_hidden_layers=6,
        attention_dropout=0.0,
        residual_dropout=0.1,
        relu_dropout=0.0,
        label_smoothing=0.1,
        fix_embedding=False,
        layer_preprocessor="none",
        layer_postprocessor="layer_norm",
        attention_key_channels=None,
        attention_value_channels=None,
        attention_function="dot_product",
        layer_type="ffn_layer",
        multiply_embedding_mode="sqrt_depth",
        pos="timing"
    )

    return params


def _residual_fn(x, y, params):
    if params.residual_dropout > 0.0:
        y = tf.nn.dropout(y, 1.0 - params.residual_dropout)

    return layer_norm(x + y)


def _dynamic_rnn(cell, inputs, sequence_length, direction, time_major=False,
                 parallel_iterations=None, swap_memory=True, dtype=None):
    if time_major:
        batch_axis = 1
        seq_axis = 0
    else:
        batch_axis = 0
        seq_axis = 1

    if direction == "backward":
        inputs = tf.reverse_sequence(inputs, sequence_length,
                                     seq_axis=seq_axis, batch_axis=batch_axis)

    outputs, final_state = tf.nn.dynamic_rnn(
        cell,
        inputs,
        sequence_length=sequence_length,
        initial_state=None,
        dtype=dtype or inputs.dtype,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        time_major=time_major,
        scope=direction
    )

    if direction == "backward":
        outputs = tf.reverse_sequence(outputs, sequence_length,
                                      seq_axis=seq_axis, batch_axis=batch_axis)

    return outputs


def _rnn_layer(inputs, seq_len, hidden_size):
    with tf.variable_scope("recurrent"):
        x = inputs
        with tf.variable_scope("forward"):
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            y_fw = _dynamic_rnn(cell_fw, x, seq_len, "forward")

        with tf.variable_scope("backward"):
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            y_bw = _dynamic_rnn(cell_bw, x, seq_len, "backward")

        return y_fw + y_bw


def _cnn_layer(inputs, mask, hidden_size, filter_width):
    with tf.variable_scope("convolution"):
        input_size = inputs.get_shape().as_list()[-1]
        shape = [filter_width, input_size, 2 * hidden_size]
        filter_v = tf.get_variable("filter", shape)
        bias_v = tf.get_variable("bias", [2 * hidden_size])
        output = tf.nn.convolution(inputs, filter_v, "SAME")
        output = tf.nn.bias_add(output, bias_v)
        gate, act = tf.split(output, 2, 2)
        output = tf.nn.sigmoid(gate) * act

        return output * mask[:, :, None]


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
               data_format="NHWC", dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = linear(inputs, hidden_size, True, data_format=data_format)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = linear(hidden, output_size, True, data_format=data_format)

        return output


def encoder(encoder_input, mask, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[encoder_input, mask]):
        x = encoder_input
        seq_len = tf.to_int32(tf.reduce_sum(mask, -1))
        attn_bias = ops.attention.attention_bias(mask, "masking")

        for layer in xrange(params.num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("computation"):
                    if params.layer_type == "ffn_layer":
                        y = _ffn_layer(
                            x,
                            params.filter_size,
                            params.hidden_size,
                            1.0 - params.relu_dropout,
                        )
                        x = _residual_fn(x, y, params)
                    elif params.layer_type == "rnn_layer":
                        y = _rnn_layer(
                            x,
                            seq_len,
                            params.hidden_size
                        )
                        x = _residual_fn(x, y, params)
                    elif params.layer_type == "cnn_layer":
                        y = _cnn_layer(
                            x,
                            mask,
                            params.hidden_size,
                            params.filter_width
                        )
                        x = _residual_fn(x, y, params)

                    # Do not use non-linear layer otherwise

                with tf.variable_scope("self_attention"):
                    y = multihead_attention(
                        x,
                        None,
                        attn_bias,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        params.num_heads,
                        1.0 - params.attention_dropout,
                        attention_function=params.attention_function
                    )
                    x = _residual_fn(x, y, params)

        return x


def deepatt_model(features, mode, params):
    hparams = params
    params = copy.copy(hparams)

    # disable dropout in evaluation/inference mode
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
        params.attention_dropout = 0.0
        params.residual_dropout = 0.0
        params.relu_dropout = 0.0

    vocab_size = len(params.vocabulary["inputs"])
    label_size = len(params.vocabulary["targets"])
    hidden_size = params.hidden_size
    feature_size = params.feature_size

    tok_seq = features["inputs"]
    pred_seq = features["preds"]
    mask = tf.to_float(tf.not_equal(tok_seq, 0))

    # shared embedding and softmax weights
    initializer = None

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        if not params.use_global_initializer:
            initializer = tf.random_normal_initializer(0.0,
                                                       feature_size ** -0.5)

    weights = tf.get_variable("weights", [2, feature_size],
                              initializer=initializer)

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        if params.embedding is not None:
            initializer = lambda shape, dtype, partition_info: params.embedding
    else:
        initializer = None

    embedding = tf.get_variable("embedding", [vocab_size, feature_size],
                                initializer=initializer,
                                trainable=not params.fix_embedding)
    bias = tf.get_variable("bias", [hidden_size])

    # id => embedding
    # src_seq: [batch, max_src_length]
    # tgt_seq: [batch, max_tgt_length]
    inputs = tf.gather(embedding, tok_seq)

    if mode == tf.contrib.learn.ModeKeys.INFER:
        if features.get("mask") is not None:
            keep_mask = features["mask"][:, :, None]
            unk_emb = features["embedding"]
            inputs = inputs * keep_mask + (1.0 - keep_mask) * unk_emb

    preds = tf.gather(weights, pred_seq)
    inputs = tf.concat([inputs, preds], -1)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(mask, -1)

    # preparing encoder & decoder input
    encoder_input = tf.nn.bias_add(inputs, bias)

    if params.pos == "timing":
        encoder_input = ops.attention.add_timing_signal(encoder_input)
    elif params.pos == "embedding":
        initializer = tf.random_normal_initializer(0.0, hidden_size ** -0.5)
        embedding = tf.get_variable("position_embedding", [1000, hidden_size],
                                    initializer=initializer)
        indices = tf.range(tf.shape(features["inputs"])[1])[None, :]
        pos_emb = tf.gather(embedding, indices)
        pos_emb = tf.tile(pos_emb, [tf.shape(features["inputs"])[0], 1, 1])
        encoder_input = encoder_input + pos_emb

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    encoder_output = encoder(encoder_input, mask, params)

    initializer = None

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        if not params.use_global_initializer:
            initializer = tf.random_normal_initializer(0.0,
                                                       hidden_size ** -0.5)

    with tf.variable_scope("prediction", initializer=initializer):
        logits = linear(encoder_output, label_size, True, scope="logits")

    if mode == tf.contrib.learn.ModeKeys.INFER:
        outputs = tf.to_int32(tf.argmax(logits, axis=-1))
        return outputs, tf.nn.softmax(logits)

    labels = features["targets"]
    targets = features["targets"]
    logits = tf.reshape(logits, [-1, label_size])
    labels = tf.reshape(labels, [-1])

    # label smoothing
    ce = ops.layers.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        label_smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(targets))
    cost = tf.reduce_sum(ce * mask) / tf.reduce_sum(mask)

    # greedy decoding
    if mode == tf.contrib.learn.ModeKeys.EVAL:
        outputs = tf.to_int32(tf.argmax(logits, axis=-1))
        return cost, tf.reshape(outputs, tf.shape(targets))

    return cost
