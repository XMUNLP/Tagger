# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import six
import time
import torch

import tagger.data as data
import tagger.models as models
import tagger.utils as utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict using SRL models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True,
                        help="Path of input file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    parser.add_argument("--half", action="store_true",
                        help="Use half precision for decoding")

    return parser.parse_args()


def default_params():
    params = utils.HParams(
        input=None,
        output=None,
        vocabulary=None,
        embedding="",
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        device=0,
        decode_batch_size=128
    )

    return params


def merge_params(params1, params2):
    params = utils.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not os.path.exists(m_name):
        return params

    with open(m_name) as fd:
        logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_params(params, args):
    params.parse(args.parameters)

    src_vocab, src_w2idx, src_idx2w = data.load_vocabulary(args.vocabulary[0])
    tgt_vocab, tgt_w2idx, tgt_idx2w = data.load_vocabulary(args.vocabulary[1])

    params.vocabulary = {
        "source": src_vocab, "target": tgt_vocab
    }
    params.lookup = {
        "source": src_w2idx, "target": tgt_w2idx
    }
    params.mapping = {
        "source": src_idx2w, "target": tgt_idx2w
    }

    return params


def convert_to_string(inputs, tensor, params):
    inputs = torch.squeeze(inputs)
    inputs = inputs.tolist()
    tensor = torch.squeeze(tensor, dim=1)
    tensor = tensor.tolist()
    decoded = []

    for wids, lids in zip(inputs, tensor):
        output = []
        for wid, lid in zip(wids, lids):
            if wid == 0:
                break
            output.append(params.mapping["target"][lid])
        decoded.append(b" ".join(output))

    return decoded


def main(args):
    # Load configs
    model_cls = models.get_model(args.model)
    params = default_params()
    params = merge_params(params, model_cls.default_params())
    params = import_params(args.checkpoint, args.model, params)
    params = override_params(params, args)
    torch.cuda.set_device(params.device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Create model
    with torch.no_grad():
        model = model_cls(params).cuda()

        if args.half:
            model = model.half()
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        model.eval()
        model.load_state_dict(
            torch.load(utils.best_checkpoint(args.checkpoint),
                       map_location="cpu")["model"])

        # Decoding
        dataset = data.get_dataset(args.input, "infer", params)
        fd = open(args.output, "wb")
        counter = 0

        if params.embedding is not None:
            embedding = data.load_glove_embedding(params.embedding)
        else:
            embedding = None

        for features in dataset:
            t = time.time()
            counter += 1
            features = data.lookup(features, "infer", params, embedding)

            labels = model.argmax_decode(features)
            batch = convert_to_string(features["inputs"], labels, params)

            for seq in batch:
                fd.write(seq)
                fd.write(b"\n")

            t = time.time() - t
            print("Finished batch: %d (%.3f sec)" % (counter, t))

        fd.close()


if __name__ == "__main__":
    main(parse_args())
