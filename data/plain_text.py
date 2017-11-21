# srl_plain_text.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import operator
import numpy as np
import tensorflow as tf


def _get_sorted_inputs(filename):
    decode_filename = filename

    # read file
    with tf.gfile.Open(decode_filename) as fd:
        inputs = [line.strip().split("|||")[0] for line in fd]

    input_lens = [
        (i, len(line.strip().split())) for i, line in enumerate(inputs)
    ]
    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))

    sorted_keys = {}
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i

    return sorted_inputs, sorted_keys


def _decode_batch_input_fn(num_decode_batches, sorted_inputs,
                           batch_size, preprocess_fn):
    # First reverse all the input sentences so that if you're going to get
    # OOMs, you'll see it in the first batch
    sorted_inputs.reverse()

    for b in range(num_decode_batches):
        batch_length = 0
        batch_inputs = []

        for inputs in sorted_inputs[b*batch_size:(b+1)*batch_size]:
            outputs = preprocess_fn(inputs)
            batch_inputs.append(list(outputs))

            if len(outputs[0]) > batch_length:
                batch_length = len(outputs[0])

        final_batch_inputs = []
        final_batch_preds = []
        final_batch_emb = []
        final_batch_mask = []

        # pad zeros
        for item in batch_inputs:
            if len(item) == 2:
                input_ids, pred_id = item
                emb, mask = None, None
            else:
                input_ids, pred_id, emb, mask = item

            assert len(input_ids) <= batch_length
            x = input_ids + [0] * (batch_length - len(input_ids))
            y = [0 for _ in x]
            y[pred_id] = 1
            final_batch_inputs.append(x)
            final_batch_preds.append(y)

            if emb is not None:
                dim = emb.shape[1]
                new_emb = np.zeros([batch_length, dim])
                new_emb[:len(input_ids), :] = emb
                final_batch_emb.append(new_emb)

            if mask is not None:
                mask = mask + [1] * (batch_length - len(input_ids))
                final_batch_mask.append(mask)

        if not final_batch_emb:
            features = {
                "inputs": np.array(final_batch_inputs),
                "preds": np.array(final_batch_preds)
            }
        else:
            features = {
                "inputs": np.array(final_batch_inputs),
                "preds": np.array(final_batch_preds),
                "embedding": np.array(final_batch_emb),
                "mask": np.array(final_batch_mask)
            }

        yield features


def load_vocab(filename):
    fd = open(filename, "r")

    count = 0
    vocab = {}
    for line in fd:
        word = line.strip()
        vocab[word] = count
        count += 1

    fd.close()

    return vocab


def load_glove_embedding(filename, vocab):
    fd = open(filename, "r")
    emb = {}
    fan_out = 0

    for line in fd:
        items = line.strip().split()
        word = items[0]
        value = [float(item) for item in items[1:]]
        fan_out = len(value)
        emb[word] = np.array(value, "float32")

    if not vocab:
        return emb

    ivoc = {}

    for item in vocab:
        ivoc[vocab[item]] = item

    new_emb = np.zeros([len(ivoc), fan_out], "float32")

    for i in ivoc:
        word = ivoc[i]
        if word not in emb:
            fan_in = len(ivoc)
            scale = 3.0 / max(1.0, (fan_in + fan_out) / 2.0)
            new_emb[i] = np.random.uniform(-scale, scale, [fan_out])
        else:
            new_emb[i] = emb[word]

    return new_emb


def convert_text(text, vocab, params):
    words = text.strip().split()
    unk = vocab["<unk>"]
    tokens = [word.lower() for word in words[1:]]
    ids = [vocab[word] if word in vocab else unk for word in tokens]

    if params.embedding is None:
        return ids, int(words[0])

    keep_mask = []
    emb = np.zeros([len(words[1:]), params.feature_size])

    for i, word in enumerate(words[1:]):
        if word in params.embedding:
            emb[i] = params.embedding[word]

        if word not in vocab and word in params.embedding:
            # drop
            keep_mask.append(0)
        else:
            # keep
            keep_mask.append(1)

    return ids, int(words[0]), emb, keep_mask


def get_sorted_input_fn(filename, vocab, batch_size, params):
    sorted_inputs, sorted_keys = _get_sorted_inputs(filename)
    num_decode_batches = (len(sorted_inputs) - 1) // batch_size + 1
    input_fn = _decode_batch_input_fn(num_decode_batches,
                                      sorted_inputs, batch_size,
                                      lambda x: convert_text(x, vocab, params))

    return sorted_inputs, sorted_keys, num_decode_batches, input_fn
