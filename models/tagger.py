# tagger.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf

from deepatt import deepatt_default_params, deepatt_model


def get_model_params(name):
    if name == "deepatt":
        return deepatt_default_params()
    else:
        raise ValueError("Unknown model name: %s" % name)


def get_tagger_model(name, mode=tf.contrib.learn.ModeKeys.TRAIN):
    if name == "deepatt":
        return lambda f, p: deepatt_model(f, mode, p)
    else:
        raise ValueError("Unknown model name: %s" % name)
