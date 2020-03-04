# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def load_glove_embedding(filename, vocab=None):
    fd = open(filename, "r")
    emb = {}
    fan_out = 0

    for line in fd:
        items = line.strip().split()
        word = items[0].encode("utf-8")
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
