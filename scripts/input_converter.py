# input_converter.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn


import os
import six
import json
import random
import argparse
import tensorflow as tf


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


def to_json(dictionary):
    """ Convert python dictionary to JSON format """
    return json.dumps(dictionary)


def to_dictionary(example):
    """ Convert JSON/tf.train.Example to python dictionary """
    if isinstance(example, str):
        dictionary = json.loads(example)
    elif isinstance(example, tf.train.Example):
        dictionary = {}
        keys = example.features.feature.keys()
        values = example.features.feature.values()

        for (k, v) in zip(keys, values):
            int64_list = list(v.int64_list.value)
            float_list = list(v.float_list.value)
            bytes_list = list(v.bytes_list.value)

            if int64_list:
                dictionary[k] = int64_list
            elif float_list:
                dictionary[k] = float_list
            elif bytes_list:
                dictionary[k] = bytes_list
            else:
                raise ValueError("All lists are empty.")
    else:
        raise ValueError("Unsupported format")

    return dictionary


def to_example(dictionary):
    """ Convert python dictionary to tf.train.Example """
    features = {}

    for (k, v) in six.iteritems(dictionary):
        if not v:
            raise ValueError("Empty generated field: %s", str((k, v)))

        if isinstance(v[0], six.integer_types):
            int64_list = tf.train.Int64List(value=v)
            features[k] = tf.train.Feature(int64_list=int64_list)
        elif isinstance(v[0], float):
            float_list = tf.train.FloatList(value=v)
            features[k] = tf.train.Feature(float_list=float_list)
        elif isinstance(v[0], six.string_types):
            bytes_list = tf.train.BytesList(value=v)
            features[k] = tf.train.Feature(bytes_list=bytes_list)
        else:
            raise ValueError("Value is neither an int nor a float; "
                             "v: %s type: %s" % (str(v[0]), str(type(v[0]))))

    return tf.train.Example(features=tf.train.Features(feature=features))


def read_records(filename):
    """ Read TensorFlow record """
    reader = tf.python_io.tf_record_iterator(filename)
    records = []

    for record in reader:
        records.append(record)
        if len(records) % 10000 == 0:
            tf.logging.info("read: %d", len(records))

    return records


def write_records(records, out_filename):
    """ Write to TensorFlow record """
    writer = tf.python_io.TFRecordWriter(out_filename)

    for count, record in enumerate(records):
        writer.write(record)
        if count % 10000 == 0:
            tf.logging.info("write: %d", count)

    writer.close()


def convert_record_to_json(pattern, output_name, output_dir, num_shards=1):
    """ Convert TensorFlow record to JSON format """
    output_files = []
    writers = []

    for shard in xrange(num_shards):
        output_filename = "%s-%.5d-of-%.5d" % (output_name, shard, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        output_files.append(output_file)
        writers.append(tf.gfile.GFile(output_file, "w"))

    filenames = tf.gfile.Glob(pattern)
    records = []

    for filename in filenames:
        records.extend(read_records(filename))

    counter, shard = 0, 0

    for record in records:
        counter += 1
        example = tf.train.Example()
        example.ParseFromString(record)
        features = to_dictionary(example)
        json_str = to_json(features)
        writers[shard].write(json_str + "\n")
        shard = (shard + 1) % num_shards

    for writer in writers:
        writer.close()


# format:
# pred-pos tokens ||| labels
def convert_plain_to_json(name, vocabs, output_name, output_dir, num_shards,
                          lower=True, shuffle=True):
    """ Convert plain SRL data to TensorFlow record """
    vocab_token = load_vocab(vocabs[0])
    vocab_label = load_vocab(vocabs[1])
    records = []
    unk = vocab_token["<unk>"]

    with open(name) as fd:
        for line in fd:
            features, labels = line.strip().split("|||")
            features = features.strip().split(" ")
            labels = labels.strip().split(" ")
            pred_pos = features[0]
            inputs = features[1:]

            if lower:
                inputs = [item.lower() for item in inputs]

            inputs = [vocab_token[item] if item in vocab_token else unk
                      for item in inputs]
            labels = [vocab_label[item] for item in labels]
            preds = [0 for _ in inputs]
            preds[int(pred_pos)] = 1

            feature = {
                "inputs": inputs,
                "preds": preds,
                "targets": labels
            }
            records.append(feature)

    if shuffle:
        random.shuffle(records)

    writers = []
    output_files = []

    for shard in xrange(num_shards):
        output_filename = "%s-%.5d-of-%.5d" % (output_name, shard, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        output_files.append(output_file)
        writers.append(tf.gfile.GFile(output_file, "w"))

    counter, shard = 0, 0

    for record in records:
        counter += 1
        features = record
        json_str = to_json(features)
        writers[shard].write(json_str + "\n")
        shard = (shard + 1) % num_shards

    for writer in writers:
        writer.close()


# format:
# pred-pos tokens ||| labels
def convert_plain_to_record(name, vocabs, output_name, output_dir, num_shards,
                            lower=True, shuffle=True):
    """ Convert plain SRL data to TensorFlow record """
    vocab_token = load_vocab(vocabs[0])
    vocab_label = load_vocab(vocabs[1])
    records = []
    unk = vocab_token["<unk>"]

    with open(name) as fd:
        for line in fd:
            features, labels = line.strip().split("|||")
            features = features.strip().split()
            labels = labels.strip().split()
            pred_pos = features[0]
            inputs = features[1:]

            if lower:
                inputs = [item.lower() for item in inputs]

            inputs = [vocab_token[item] if item in vocab_token else unk
                      for item in inputs]
            labels = [vocab_label[item] for item in labels]
            preds = [0 for _ in inputs]
            preds[int(pred_pos)] = 1

            feature = {
                "inputs": inputs,
                "preds": preds,
                "targets": labels
            }
            records.append(feature)

    if shuffle:
        random.shuffle(records)

    output_files = []
    writers = []

    for shard in xrange(num_shards):
        output_filename = "%s-%.5d-of-%.5d" % (output_name, shard, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        output_files.append(output_file)
        writers.append(tf.python_io.TFRecordWriter(output_file))

    counter, shard = 0, 0

    for record in records:
        counter += 1
        example = to_example(record)
        writers[shard].write(example.SerializeToString())
        shard = (shard + 1) % num_shards

    for writer in writers:
        writer.close()


def parse_args():
    msg = "convert srl data to TensorFlow record format"
    usage = "srl_input_converter.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "path of source file"
    parser.add_argument("--input_path", required=True, type=str, help=msg)
    msg = "output name"
    parser.add_argument("--output_name", required=True, type=str, help=msg)
    msg = "output directory"
    parser.add_argument("--output_dir", required=True, type=str, help=msg)
    msg = "path of vocabulary"
    parser.add_argument("--vocab", type=str, nargs=2, help=msg)
    msg = "number of output shards"
    parser.add_argument("--num_shards", default=100, type=int, help=msg)
    msg = "shuffle inputs"
    parser.add_argument("--shuffle", action="store_true", help=msg)
    msg = "use lowercase"
    parser.add_argument("--lower", action="store_true", help=msg)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    convert_plain_to_record(args.input_path, args.vocab, args.output_name,
                            args.output_dir, args.num_shards, args.lower,
                            args.shuffle)
