# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np


def _lookup(x, vocab, embedding=None, feature_size=0):
    x = x.tolist()
    y = []
    unk_mask = []
    embeddings = []

    for _, batch in enumerate(x):
        ids = []
        mask = []
        emb = []

        for _, v in enumerate(batch):
            if v in vocab:
                ids.append(vocab[v])
                mask.append(1.0)

                if embedding is not None:
                    emb.append(np.zeros([feature_size]))
            else:
                ids.append(2)

                if embedding is not None and v in embedding:
                    mask.append(0.0)
                    emb.append(embedding[v])
                else:
                    mask.append(1.0)
                    emb.append(np.zeros([feature_size]))

        y.append(ids)
        unk_mask.append(mask)
        embeddings.append(emb)

    ids = torch.LongTensor(np.array(y, dtype="int32")).cuda()
    mask = torch.Tensor(np.array(unk_mask, dtype="float32")).cuda()

    if embedding is not None:
        emb = torch.Tensor(np.array(embeddings, dtype="float32")).cuda()
    else:
        emb = None

    return ids, mask, emb


def load_vocabulary(filename):
    vocab = []
    with open(filename, "rb") as fd:
        for line in fd:
            vocab.append(line.strip())

    word2idx = {}
    idx2word = {}

    for idx, word in enumerate(vocab):
        word2idx[word] = idx
        idx2word[idx] = word

    return vocab, word2idx, idx2word


def lookup(inputs, mode, params, embedding=None):
    if mode == "train":
        features, labels = inputs
        preds, seqs = features["preds"], features["inputs"]
        preds = torch.LongTensor(preds.numpy()).cuda()
        seqs = seqs.numpy()
        labels = labels.numpy()

        seqs, _, _ = _lookup(seqs, params.lookup["source"])
        labels, _, _ = _lookup(labels, params.lookup["target"])

        features = {
            "preds": preds,
            "inputs": seqs
        }

        return features, labels
    else:
        features, _ = inputs
        preds, seqs = features["preds"], features["inputs"]
        preds = torch.LongTensor(preds.numpy()).cuda()
        seqs = seqs.numpy()

        seqs, unk_mask, emb = _lookup(seqs, params.lookup["source"], embedding,
                                      params.feature_size)

        features = {
            "preds": preds,
            "inputs": seqs,
            "mask": unk_mask
        }

        if emb is not None:
            features["embedding"] = emb

        return features
