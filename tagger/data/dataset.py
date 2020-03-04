# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import queue
import torch
import threading
import tensorflow as tf


_QUEUE = None
_THREAD = None
_LOCK = threading.Lock()


def build_input_fn(filename, mode, params):
    def train_input_fn():
        dataset = tf.data.TextLineDataset(filename)
        dataset = dataset.prefetch(params.buffer_size)
        dataset = dataset.shuffle(params.buffer_size)

        # Split "|||"
        dataset = dataset.map(
            lambda x: tf.strings.split([x], sep="|||", maxsplit=2),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x: (x.values[0], x.values[1]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x, y: (tf.strings.split([x]).values,
                          tf.strings.split([y]).values),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x, y: ({
                "preds": tf.strings.to_number(x[0], tf.int32),
                "inputs": tf.strings.lower(x[1:])
            }, y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x, y: ({
                "preds": tf.one_hot(x["preds"], tf.shape(x["inputs"])[0],
                                    dtype=tf.int32),
                "inputs": x["inputs"]
            }, y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def bucket_boundaries(max_length, min_length=8, step=8):
            x = min_length
            boundaries = []

            while x <= max_length:
                boundaries.append(x + 1)
                x += step

            return boundaries

        batch_size = params.batch_size
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(1, batch_size // (x - 1))
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        def element_length_func(x, y):
            return tf.shape(x["inputs"])[0]

        def valid_size(x, y):
            size = element_length_func(x, y)
            return tf.logical_and(size >= min_length, size <= max_length)

        transformation_fn = tf.data.experimental.bucket_by_sequence_length(
            element_length_func,
            boundaries,
            batch_sizes,
            padded_shapes=({
                "inputs": tf.TensorShape([None]),
                "preds": tf.TensorShape([None]),
                }, tf.TensorShape([None])),
            padding_values=({
                "inputs": params.pad,
                "preds": 0,
                }, params.pad),
            pad_to_bucket_boundary=True)

        dataset = dataset.filter(valid_size)
        dataset = dataset.apply(transformation_fn)

        return dataset


    def infer_input_fn():
        dataset = tf.data.TextLineDataset(filename)

        # Split "|||"
        dataset = dataset.map(
            lambda x: tf.strings.split([x], sep="|||", maxsplit=2),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x: (x.values[0], x.values[1]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x, y: (tf.strings.split([x]).values,
                          tf.strings.split([y]).values),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x, y: ({
                "preds": tf.strings.to_number(x[0], tf.int32),
                "inputs": tf.strings.lower(x[1:])
            }, y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x, y: ({
                "preds": tf.one_hot(x["preds"], tf.shape(x["inputs"])[0],
                                    dtype=tf.int32),
                "inputs": x["inputs"]
            }, y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.padded_batch(
            params.decode_batch_size,
            padded_shapes=({
                "inputs": tf.TensorShape([None]),
                "preds": tf.TensorShape([None]),
                }, tf.TensorShape([None])),
            padding_values=({
                "inputs": params.pad,
                "preds": 0,
                }, params.pad),
            )

        return dataset

    if mode == "train":
        return train_input_fn
    else:
        return infer_input_fn


class DatasetWorker(threading.Thread):

    def init(self, dataset):
        self._dataset = dataset
        self._stop = False

    def run(self):
        global _QUEUE
        global _LOCK

        while not self._stop:
            for feature in self._dataset:
                _QUEUE.put(feature)

    def stop(self):
        self._stop = True


class Dataset(object):

    def __iter__(self):
        return self

    def __next__(self):
        global _QUEUE
        return _QUEUE.get()

    def stop(self):
        global _THREAD
        _THREAD.stop()
        _THREAD.join()


def get_dataset(filenames, mode, params):
    global _QUEUE
    global _THREAD

    input_fn = build_input_fn(filenames, mode, params)

    with tf.device("/cpu:0"):
        dataset = input_fn()

    if mode != "train":
        return dataset
    else:
        _QUEUE = queue.Queue(100)
        thread = DatasetWorker(daemon=True)
        thread.init(dataset)
        thread.start()
        _THREAD = thread
        return Dataset()
