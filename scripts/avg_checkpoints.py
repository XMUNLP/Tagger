import six
import argparse
import numpy as np
from six.moves import zip
import tensorflow as tf


def parseargs():
    msg = "average checkpoints"
    usage = "average.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "name of checkpoints"
    parser.add_argument("--checkpoints", nargs="+", type=str, help=msg)
    msg = "output name"
    parser.add_argument("--output_path", default="avg.ckpt", type=str, help=msg)

    return parser.parse_args()


def checkpoint_exists(path):
    return (tf.gfile.Exists(path) or tf.gfile.Exists(path + ".meta") or
               tf.gfile.Exists(path + ".index"))


def main(args):
    # Get the checkpoints list from flags and run some basic checks.
    checkpoints = [c.strip() for c in args.checkpoints]
    checkpoints = [c for c in checkpoints if c]

    if not checkpoints:
        raise ValueError("No checkpoints provided for averaging.")

    checkpoints = [c for c in checkpoints if checkpoint_exists(c)]

    if not checkpoints:
        raise ValueError(
            "None of the provided checkpoints exist. %s" % args.checkpoints
        )

    var_list = tf.contrib.framework.list_variables(checkpoints[0])
    var_values, var_dtypes = {}, {}

    for (name, shape) in var_list:
        if not name.startswith("global_step"):
            var_values[name] = np.zeros(shape)

    for checkpoint in checkpoints:
        reader = tf.contrib.framework.load_checkpoint(checkpoint)
        for name in var_values:
            tensor = reader.get_tensor(name)
            var_dtypes[name] = tensor.dtype
            var_values[name] += tensor
        tf.logging.info("Read from checkpoint %s", checkpoint)

    for name in var_values:  # Average.
        var_values[name] /= len(checkpoints)

    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[name])
        for v in var_values
    ]
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    global_step = tf.Variable(0, name="global_step", trainable=False,
                              dtype=tf.int64)
    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                               six.iteritems(var_values)):
            sess.run(assign_op, {p: value})
        # Use the built saver to save the averaged checkpoint.
        saver.save(sess, args.output_path, global_step=global_step)

    tf.logging.info("Averaged checkpoints saved in %s", args.output_path)


if __name__ == "__main__":
    args = parseargs()
    main(args)
