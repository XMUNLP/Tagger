# main.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import ops
import sys
import copy
import argparse
import numpy as np
import tensorflow as tf
import multiprocessing

from utils import parallel_model
from utils.validation import validate
from data.record_reader import get_input_fn
from data.plain_text import load_vocab, load_glove_embedding
from data.plain_text import get_sorted_input_fn, convert_text
from ops.initializer import variance_scaling_initializer
from models.tagger import get_tagger_model, get_model_params
from metrics import create_tagger_evaluation_metrics


def parseargs_train(args):
    msg = "training SRL models"
    usage = "main.py train [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "path or pattern of input data"
    parser.add_argument("--data_path", type=str, help=msg)
    msg = "directory to save models"
    parser.add_argument("--model_dir", type=str, help=msg)
    msg = "name of model"
    parser.add_argument("--model_name", type=str, help=msg)
    msg = "path to token and label vocabulary"
    parser.add_argument("--vocab_path", type=str, nargs=2, help=msg)
    msg = "pre-trained embedding file"
    parser.add_argument("--emb_path", type=str, help=msg)
    msg = "model parameters, see tf.contrib.training.parse_values for details"
    parser.add_argument("--model_params", default="", type=str, help=msg)
    msg = "training parameters"
    parser.add_argument("--training_params", default="", type=str, help=msg)
    msg = "validation params"
    parser.add_argument("--validation_params", default="", type=str, help=msg)
    msg = "decoding parameters"
    parser.add_argument("--decoding_params", default="", type=str, help=msg)

    return parser.parse_args(args)


def parseargs_predict(args):
    msg = "predict using existing SRL models"
    usage = "main.py predict [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "path or pattern of input data"
    parser.add_argument("--data_path", type=str, help=msg)
    msg = "directory to save models"
    parser.add_argument("--model_dir", type=str, help=msg)
    msg = "name of model"
    parser.add_argument("--model_name", type=str, help=msg)
    msg = "name of output file"
    parser.add_argument("--output_name", type=str, help=msg)
    msg = "path to token and label vocabulary"
    parser.add_argument("--vocab_path", type=str, nargs=2, help=msg)
    msg = "pretrained embedding path"
    parser.add_argument("--emb_path", type=str, help=msg)
    msg = "model parameters, see tf.contrib.training.parse_values for details"
    parser.add_argument("--model_params", default="", type=str, help=msg)
    msg = "decoding parameters"
    parser.add_argument("--decoding_params", default="", type=str, help=msg)
    msg = "use viterbi decoding"
    parser.add_argument("--viterbi", action="store_true", help=msg)
    msg = "enable verbose message"
    parser.add_argument("--verbose", action="store_true", help=msg)
    msg = "decoding device"
    parser.add_argument("--device_list", nargs="+", type=int, help=msg)

    return parser.parse_args(args)


def parseargs_ensemble(args):
    msg = "ensemble using existing SRL models"
    usage = "main.py ensemble [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "path or pattern of input data"
    parser.add_argument("--data_path", type=str, help=msg)
    msg = "directory to save models"
    parser.add_argument("--checkpoints", nargs="+", type=str, help=msg)
    msg = "name of model"
    parser.add_argument("--model_name", type=str, help=msg)
    msg = "name of output file"
    parser.add_argument("--output_name", type=str, help=msg)
    msg = "path to token and label vocabulary"
    parser.add_argument("--vocab_path", type=str, nargs="+", help=msg)
    msg = "pretrained embedding path"
    parser.add_argument("--emb_path", type=str, help=msg)
    msg = "model parameters, see tf.contrib.training.parse_values for details"
    parser.add_argument("--model_params", default="", type=str, help=msg)
    msg = "decoding parameters"
    parser.add_argument("--decoding_params", default="", type=str, help=msg)
    msg = "use viterbi decoding"
    parser.add_argument("--viterbi", action="store_true", help=msg)
    msg = "enable verbose message"
    parser.add_argument("--verbose", action="store_true", help=msg)
    msg = "decoding device"
    parser.add_argument("--device_list", nargs="+", type=int, help=msg)

    return parser.parse_args(args)


def parseargs_visualize(args):
    msg = "Visualize attention weights using existing SRL models"
    usage = "main.py visualize [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "path or pattern of input data"
    parser.add_argument("--data_path", type=str, help=msg)
    msg = "directory to save models"
    parser.add_argument("--model_dir", type=str, help=msg)
    msg = "name of model"
    parser.add_argument("--model_name", type=str, help=msg)
    msg = "name of output html file"
    parser.add_argument("--output_name", type=str, help=msg)
    msg = "path to token and label vocabulary"
    parser.add_argument("--vocab_path", type=str, nargs=2, help=msg)
    msg = "pretrained embedding path"
    parser.add_argument("--emb_path", type=str, help=msg)
    msg = "model parameters, see tf.contrib.training.parse_values for details"
    parser.add_argument("--model_params", default="", type=str, help=msg)
    msg = "enable verbose message"
    parser.add_argument("--verbose", action="store_true", help=msg)
    msg = "decoding device"
    parser.add_argument("--device_list", nargs="+", type=int, help=msg)

    return parser.parse_args(args)


def get_vocabulary(vocab_path):
    tok_voc = load_vocab(vocab_path[0])
    lab_voc = load_vocab(vocab_path[1])
    vocabulary = {"inputs": tok_voc, "targets": lab_voc}

    return vocabulary


def get_ensemble_vocabulary(vocab_path):
    vocs = [load_vocab(item) for item in vocab_path]
    voc_list = []

    tok_voc = vocs[:-1]
    lab_voc = vocs[-1]

    for item in tok_voc:
        vocab = {"inputs": item, "targets": lab_voc}
        voc_list.append(vocab)

    return voc_list


def training_params():
    params = tf.contrib.training.HParams(
        optimizer="Adam",
        learning_rate=1.0,
        max_learning_rate=5e-4,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-9,
        adadelta_rho=0.95,
        adadelta_epsilon=1e-6,
        initializer_gain=1.0,
        clip_grad_norm=0.0,
        batch_size=4096,
        eval_batch_size=4096,
        max_length=256,
        mantissa_bits=2,
        warmup_steps=4000,
        train_steps=250000,
        eval_steps=10,
        min_eval_frequency=2000,
        keep_checkpoint_max=20,
        batching_scheme="token",
        learning_rate_decay="noam",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        initializer="uniform_unit_scaling",
        device_list=[0],
        allow_growth=True,
        use_global_initializer=True
    )

    return params


def validation_params():
    params = tf.contrib.training.HParams(
        script="",
        frequency=300,
        keep_top_k=5
    )

    return params


def decoding_params():
    params = tf.contrib.training.HParams(
        decode_batch_size=128,
    )

    return params


def merge_params(p1, p2):
    params = tf.contrib.training.HParams()
    v1 = p1.values()
    v2 = p2.values()

    for (k, v) in v1.iteritems():
        params.add_hparam(k, v)

    for (k, v) in v2.iteritems():
        params.add_hparam(k, v)

    return params


def get_params(args):
    params = tf.contrib.training.HParams(
        data_path=args.data_path,
        model_dir=args.model_dir,
        model_name=args.model_name,
        vocab_path=args.vocab_path,
        model_params=args.model_params,
        training_params=args.training_params
    )

    tparams = training_params()
    tparams.parse(args.training_params)
    params = merge_params(params, tparams)

    mparams = get_model_params(args.model_name)
    mparams.parse(args.model_params)
    params = merge_params(params, mparams)

    vparams = validation_params()
    vparams.parse(args.validation_params)
    params = merge_params(params, vparams)

    dparams = decoding_params()
    dparams.parse(args.decoding_params)
    params = merge_params(params, dparams)

    return params


def print_params(params):
    for (k, v) in params.values():
        print("%s: %s" % (k, v))


def orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32):
    oinit = tf.orthogonal_initializer(gain, seed, dtype)

    def initializer(shape, dtype=dtype, partition_info=None):
        if len(shape) == 1:
            result = oinit(list(shape) + [1], dtype, partition_info)
            return tf.squeeze(result, 1)
        return oinit(shape, dtype, partition_info)

    return initializer


def get_transition_params(label_strs):
    num_tags = len(label_strs)
    transition_params = np.zeros([num_tags, num_tags], dtype=np.float32)

    for i, prev_label in enumerate(label_strs):
        for j, label in enumerate(label_strs):
            if prev_label[0] == "B" and label[0] == "I":
                if prev_label[1:] != label[1:]:
                    transition_params[i, j] = np.NINF

            if prev_label[0] == "I" and label[0] == "I":
                if prev_label[1:] != label[1:]:
                    transition_params[i, j] = np.NINF

    return transition_params


def get_initializer(params):
    if params.initializer == "orthogonal":
        return orthogonal_initializer(gain=params.initializer_gain)
    elif params.initializer == "uniform":
        max_val = params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return variance_scaling_initializer(params.initializer_gain,
                                            mode="fan_avg",
                                            distribution="normal")
    elif params.initializer == "uniform_unit_scaling":
        return variance_scaling_initializer(params.initializer_gain,
                                            mode="fan_avg",
                                            distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)


def get_learning_rate_decay(learning_rate, global_step, params):
    if params.learning_rate_decay == "noam":
        return ops.train.noam_decay(learning_rate, global_step,
                                    params.warmup_steps,
                                    params.hidden_size ** -0.5)
    elif params.learning_rate_decay == "piecewise_constant":
        return tf.train.piecewise_constant(tf.to_int32(global_step),
                                           params.learning_rate_boundaries,
                                           params.learning_rate_values)
    elif params.learning_rate_decay == "none":
        return learning_rate
    else:
        raise ValueError("Unknown learning_rate_decay")


def set_variables(var_list, value_dict, prefix):
    sess = tf.get_default_session()

    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))
            if var.name[:-2] == var_name:
                print("restoring %s -> %s" % (name, var.name))
                with tf.device("/cpu:0"):
                    op = tf.assign(var, value_dict[name])
                    sess.run(op)
                break


def srl_model(features, labels, mode, params):
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        initializer = get_initializer(params)
        tf.get_variable_scope().set_initializer(initializer)
        model_fn = get_tagger_model(params.model_name, mode)

        features["targets"] = labels

        with tf.variable_scope("tagger"):
            loss = parallel_model(model_fn, features, params, mode)

            with tf.variable_scope("losses_avg"):
                loss_moving_avg = tf.get_variable("training_loss",
                                                  initializer=100.0,
                                                  trainable=False)
                lm = loss_moving_avg.assign(loss_moving_avg * 0.9 + loss * 0.1)
                tf.summary.scalar("loss_avg/total_loss", lm)

                with tf.control_dependencies([lm]):
                    loss = tf.identity(loss)

        global_step = tf.train.get_or_create_global_step()
        lr = get_learning_rate_decay(params.learning_rate, global_step, params)

        # create optimizer
        if params.optimizer == "Adam":
            opt = tf.train.AdamOptimizer(lr, beta1=params.adam_beta1,
                                         beta2=params.adam_beta2,
                                         epsilon=params.adam_epsilon)
        elif params.optimizer == "Adadelta":
            opt = tf.train.AdadeltaOptimizer(lr, rho=params.adadelta_rho,
                                             epsilon=params.adadelta_epsilon)
        elif params.optimizer == "SGD":
            opt = tf.train.GradientDescentOptimizer(lr)
        elif params.optimizer == "Nadam":
            opt = tf.contrib.opt.NadamOptimizer(lr, beta1=params.adam_beta1,
                                                beta2=params.adam_beta2,
                                                epsilon=params.adam_epsilon)
        else:
            raise ValueError("Unknown optimizer %s" % params.optimizer)

        global_step = tf.train.get_global_step()
        tf.summary.scalar("learning_rate", lr)

        log_hook = tf.train.LoggingTensorHook(
            {
                "step": global_step,
                "loss": loss,
                "inputs": tf.shape(features["inputs"]),
                "labels": tf.shape(labels)
            },
            every_n_iter=1,
        )

        all_weights = {v.name: v for v in tf.trainable_variables()}
        total_size = 0

        for v_name in sorted(list(all_weights)):
            v = all_weights[v_name]
            tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                            str(v.shape).ljust(20))
            v_size = int(np.prod(np.array(v.shape.as_list())))
            total_size += v_size

        tf.logging.info("Total trainable variables size: %d", total_size)

        train_op = tf.contrib.layers.optimize_loss(
            name="training",
            loss=loss,
            global_step=global_step,
            learning_rate=lr,
            clip_gradients=params.clip_grad_norm or None,
            optimizer=opt,
            colocate_gradients_with_ops=True
        )

        training_chief_hooks = [log_hook]
        predictions = None
    elif mode == tf.contrib.learn.ModeKeys.EVAL:
        model_fn = get_tagger_model(params.model_name, mode)
        features["targets"] = labels
        with tf.variable_scope("tagger"):
            loss, logits = model_fn(features, params)
        predictions = {"predictions": logits}
        train_op = None
        training_chief_hooks = None
    elif mode == tf.contrib.learn.ModeKeys.INFER:
        model_fn = get_tagger_model(params.model_name, mode)
        features["targets"] = labels
        with tf.variable_scope("tagger"):
            outputs, probs = model_fn(features, params)
        predictions = {
            "inputs": features["inputs"],
            "outputs": outputs,
            "distribution": probs
        }
        loss = None
        train_op = None
        training_chief_hooks = None
    else:
        raise ValueError("Unknown mode %s" % mode)

    spec = tf.contrib.learn.ModelFnOps(
        mode=mode, loss=loss, train_op=train_op,
        training_chief_hooks=training_chief_hooks,
        predictions=predictions
    )

    return spec


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    if params.allow_growth:
        config.gpu_options.allow_growth = True

    return config


def train(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    params = get_params(args)
    vocabulary = get_vocabulary(params.vocab_path)
    params.add_hparam("vocabulary", vocabulary)

    if args.emb_path:
        if args.emb_path.find("glove") > 0:
            emb = load_glove_embedding(args.emb_path,
                                       params.vocabulary["inputs"])
        else:
            emb = np.loadtxt(args.emb_path).astype("float32")
    else:
        emb = None

    params.add_hparam("embedding", emb)

    config = tf.contrib.learn.RunConfig(
        model_dir=params.model_dir,
        session_config=session_config(params),
        keep_checkpoint_max=params.keep_checkpoint_max,
        save_checkpoints_secs=300
    )

    # model_fn: (features, labels, mode, params, conifg) => EstimatorSpec
    # input_fn:  () => (features, labels)

    # create estimator
    estimator = tf.contrib.learn.Estimator(
        model_fn=srl_model,
        model_dir=params.model_dir,
        config=config,
        params=params
    )

    # create input_fn
    train_input_fn = get_input_fn(
        params.data_path + "*train*",
        tf.contrib.learn.ModeKeys.TRAIN,
        params
    )

    if tf.gfile.Glob(params.data_path + "*dev*"):
        eval_input_fn = get_input_fn(
            params.data_path + "*dev*", tf.contrib.learn.ModeKeys.EVAL, params
        )
    else:
        eval_input_fn = None

    # create experiment
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        eval_metrics=create_tagger_evaluation_metrics(),
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=params.train_steps,
        eval_steps=params.eval_steps,
        min_eval_frequency=params.min_eval_frequency
    )

    if params.script:
        process = multiprocessing.Process(target=validate, args=[params])
        process.daemon = True
        process.start()
    else:
        process = None

    # start training
    try:
        if eval_input_fn:
            experiment.train_and_evaluate()
        else:
            experiment.train()
    finally:
        if process is not None:
            process.terminate()


def predict(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    params = tf.contrib.training.HParams(
        data_path=args.data_path,
        model_dir=args.model_dir,
        model_name=args.model_name,
        vocab_path=args.vocab_path,
        model_params=args.model_params,
        device_list=args.device_list or [0],
        allow_growth=True
    )

    mparams = get_model_params(args.model_name)
    params = merge_params(params, mparams)
    params.parse(args.model_params)
    dparams = decoding_params()
    params = merge_params(params, dparams)
    params.parse(args.decoding_params)

    vocabulary = get_vocabulary(params.vocab_path)
    params.add_hparam("vocabulary", vocabulary)

    if args.emb_path:
        if args.emb_path.find("glove") > 0:
            emb = load_glove_embedding(args.emb_path, None)
        else:
            emb = np.loadtxt(args.emb_path).astype("float32")
    else:
        emb = None

    params.add_hparam("embedding", emb)

    config = tf.contrib.learn.RunConfig(
        model_dir=params.model_dir,
        session_config=session_config(params),
    )

    # create estimator
    estimator = tf.contrib.learn.Estimator(
        model_fn=srl_model,
        model_dir=params.model_dir,
        config=config,
        params=params
    )

    decodes = []
    sorted_inputs, sorted_keys, num_batches, input_fn = get_sorted_input_fn(
        params.data_path,
        params.vocabulary["inputs"],
        params.decode_batch_size * len(params.device_list),
        params
    )

    ivocab = {"inputs": {}, "targets": {}}
    labels = []

    for k, idx in vocabulary["inputs"].iteritems():
        ivocab["inputs"][idx] = k

    for k, idx in vocabulary["targets"].iteritems():
        ivocab["targets"][idx] = k

    for idx in range(len(ivocab["targets"])):
        labels.append(ivocab["targets"][idx])

    tparams = get_transition_params(labels)

    for i in range(num_batches):
        result_iter = estimator.predict(input_fn=input_fn.next,
                                        as_iterable=True)

        for result in result_iter:
            inputs = result["inputs"]
            outputs = result["outputs"]
            dist = result["distribution"]
            input_text = []
            output_text = []

            index = 0

            if args.viterbi:
                seq_len = 0
                while index < len(inputs) and inputs[index] != 0:
                    seq_len += 1
                    index += 1
                dist = dist[:seq_len, :]
                outputs, _ = tf.contrib.crf.viterbi_decode(dist, tparams)

            index = 0

            while index < len(inputs) and inputs[index] != 0:
                input_text.append(ivocab["inputs"][inputs[index]])
                output_text.append(ivocab["targets"][outputs[index]])
                index += 1

            # decode to plain text
            input_text = " ".join(input_text)
            output_text = " ".join(output_text)

            if args.verbose:
                sys.stdout.write("INPUT: %s\n" % input_text)
                sys.stdout.write("OUTPUT: %s\n" % output_text)

            decodes.append(output_text)

    sorted_inputs.reverse()
    decodes.reverse()

    outputs = []

    for index in range(len(sorted_inputs)):
        outputs.append(decodes[sorted_keys[index]])

    if not args.output_name:
        base_filename = os.path.basename(params.data_path)
        decode_filename = base_filename + "." + params.model_name + ".decodes"
    else:
        decode_filename = args.output_name

    outfile = tf.gfile.Open(decode_filename, "w")

    for output in outputs:
        outfile.write("%s\n" % output)

    outfile.close()


def ensemble(args):
    if len(args.vocab_path) != len(args.checkpoints) + 1:
        raise ValueError("Unmatched vocabulary number and checkpoint number")

    # override parameters
    params = tf.contrib.training.HParams(
        data_path=args.data_path,
        model_name=args.model_name,
        vocab_path=args.vocab_path,
        model_params=args.model_params,
        device_list=args.device_list or [0],
        allow_growth=True
    )

    mparams = get_model_params(args.model_name)
    params = merge_params(params, mparams)
    params.parse(args.model_params)
    dparams = decoding_params()
    params = merge_params(params, dparams)
    params.parse(args.decoding_params)

    if args.emb_path:
        if args.emb_path.find("glove") > 0:
            emb = load_glove_embedding(args.emb_path, None)
        else:
            emb = np.loadtxt(args.emb_path).astype("float32")
    else:
        emb = None

    vocabularies = get_ensemble_vocabulary(params.vocab_path)

    model_var_lists = []
    model_params_list = []

    for i in range(len(args.checkpoints)):
        cparams = copy.copy(params)
        cparams.add_hparam("embedding", emb)
        cparams.add_hparam("vocabulary", vocabularies[i])
        model_params_list.append(cparams)

    # load checkpoints
    for checkpoint in args.checkpoints:
        var_list = tf.train.list_variables(checkpoint)
        values = {}
        reader = tf.train.load_checkpoint(checkpoint)

        for (name, shape) in var_list:
            if not name.startswith("tagger"):
                continue

            if name.find("losses_avg") >= 0:
                continue

            tensor = reader.get_tensor(name)
            values[name] = tensor

        model_var_lists.append(values)

    # build graph
    inputs = tf.placeholder(tf.int32, [None, None], "inputs")
    preds = tf.placeholder(tf.int32, [None, None], "preds")
    embedding = tf.placeholder(tf.float32, [None, None, None], "embedding")
    mask = tf.placeholder(tf.float32, [None, None], "mask")

    features = {"inputs": inputs, "preds": preds}

    if emb is not None:
        features["embedding"] = embedding
        features["mask"] = mask

    predictions = []

    for i in range(len(args.checkpoints)):
        with tf.variable_scope("tagger_%d" % i):
            model_fn = get_tagger_model(params.model_name,
                                        tf.contrib.learn.ModeKeys.INFER)
            outputs, probs = model_fn(features, model_params_list[i])
            predictions.append(probs)

    labels = []
    ivocab = {}

    for k, idx in vocabularies[0]["targets"].iteritems():
        ivocab[idx] = k

    for idx in range(len(ivocab)):
        labels.append(ivocab[idx])

    tparams = get_transition_params(labels)

    # create session
    with tf.Session(config=session_config(params)) as sess:
        tf.global_variables_initializer().run()

        # restore variables
        all_var_list = tf.trainable_variables()

        for i in range(len(args.checkpoints)):
            uninit_var_list = []

            for v in all_var_list:
                if v.name.startswith("tagger_%d" % i):
                    uninit_var_list.append(v)

            set_variables(uninit_var_list, model_var_lists[i], "tagger_%d" % i)

        # create input_fn
        all_sorted_inputs = []
        all_sorted_keys = []
        all_input_fns = []

        for i in range(len(args.checkpoints)):
            sorted_inputs, sorted_keys, num_batches, fn = get_sorted_input_fn(
                params.data_path,
                model_params_list[i].vocabulary["inputs"],
                params.decode_batch_size * len(params.device_list),
                model_params_list[i]
            )
            all_sorted_inputs.append(sorted_inputs)
            all_sorted_keys.append(sorted_keys)
            all_input_fns.append(fn)

        decodes = []

        for i, input_fn in enumerate(all_input_fns):
            outputs = []
            for features in input_fn:
                feed_dict = {
                    inputs: features["inputs"],
                    preds: features["preds"]
                }

                if args.emb_path:
                    feed_dict[embedding] = features["embedding"]
                    feed_dict[mask] = features["mask"]

                output = sess.run(predictions[i], feed_dict=feed_dict)

                outputs.append(output)

            decodes.append(outputs)

        # ensemble
        decodes = list(zip(*decodes))
        probs = []

        for item in decodes:
            outputs = sum(item) / float(len(item))
            # [batch, max_len, num_label]
            probs.append(outputs)

        count = 0

        for item in probs:
            for dist in item:
                inputs = all_sorted_inputs[0][count]
                seq_len = len(inputs.strip().split()[1:])
                output_text = []

                if args.viterbi:
                    dist = dist[:seq_len, :]
                    outputs, _ = tf.contrib.crf.viterbi_decode(dist,
                                                               tparams)
                else:
                    dist = dist[:seq_len, :]
                    outputs = np.argmax(dist, axis=1)

                index = 0

                while index < seq_len:
                    output_text.append(ivocab[outputs[index]])
                    index += 1

                # decode to plain text
                output_text = " ".join(output_text)
                decodes.append(output_text)
                count += 1

        sorted_inputs.reverse()
        decodes.reverse()

        outputs = []

        for index in range(len(sorted_inputs)):
            outputs.append(decodes[sorted_keys[index]])

        if not args.output_name:
            base_filename = os.path.basename(params.data_path)
            model_name = params.model_name
            decode_filename = base_filename + "." + model_name + ".decodes"
        else:
            decode_filename = args.output_name

        outfile = tf.gfile.Open(decode_filename, "w")

        for output in outputs:
            outfile.write("%s\n" % output)

        outfile.close()


def helpinfo():
    print("usage:")
    print("\tmain.py <command> [<args>]")
    print("using 'main.py train --help' to see training options")
    print("using 'main.py predict --help' to see prediction options")
    print("using 'main.py ensemble --help' to see ensembling options")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        helpinfo()
    else:
        command = sys.argv[1]
        if command == "train":
            print("training command:")
            print(" ".join(sys.argv))
            parsed_args = parseargs_train(sys.argv[2:])
            train(parsed_args)
        elif command == "predict":
            parsed_args = parseargs_predict(sys.argv[2:])
            predict(parsed_args)
        elif command == "ensemble":
            parsed_args = parseargs_ensemble(sys.argv[2:])
            ensemble(parsed_args)
        else:
            helpinfo()
