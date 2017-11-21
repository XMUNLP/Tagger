# decay.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy as np
import tensorflow as tf

from tensorflow.python.training.learning_rate_decay import *

__all__ = [
    "exponential_decay",
    "inverse_time_decay",
    "natural_exp_decay",
    "piecewise_constant",
    "polynomial_decay",
    "noam_decay",
    "cosine_decay",
    "sqrt_decay"
]


def noam_decay(learning_rate, global_step, warmup_steps, multiplier):
    step = tf.to_float(global_step)
    warmup_steps = tf.to_float(warmup_steps)
    decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                    (step + 1) ** -0.5)

    return learning_rate * decay


def cosine_decay(learning_rate, global_step, cycle_steps, multiplier=0.5):
    step = tf.to_float(global_step)
    decay = 0.5 * (1 + tf.cos(np.pi * (step % cycle_steps) / cycle_steps))

    return learning_rate * decay


def sqrt_decay(learning_rate, global_step, multiplier=500.0):
    step = tf.to_float(global_step)
    decay = multiplier * tf.sqrt(tf.maximum(step, 1.0))

    return learning_rate * decay
