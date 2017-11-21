# build_vocab.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import argparse
import collections


def count_items(filename, lower=False):
    counter = collections.Counter()
    label_counter = collections.Counter()

    with open(filename, "r") as fd:
        for line in fd:
            words, labels = line.strip().split("|||")
            words = words.strip().split()
            labels = labels.strip().split()

            if lower:
                words = [item.lower() for item in words[1:]]
            else:
                words = words[1:]

            counter.update(words)
            label_counter.update(labels)

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, counts = list(zip(*count_pairs))
    count_pairs = sorted(label_counter.items(), key=lambda x: (-x[1], x[0]))
    labels, _ = list(zip(*count_pairs))

    return words, labels, counts


def special_tokens(string):
    if not string:
        return []
    else:
        return string.strip().split(":")


def save_vocab(name, vocab):
    if name.split(".")[-1] != "txt":
        name = name + ".txt"

    pairs = sorted(vocab.items(), key=lambda x: (x[1], x[0]))
    words, ids = list(zip(*pairs))

    with open(name, "w") as f:
        for word in words:
            f.write(word + "\n")


def write_vocab(name, vocab):
    with open(name, "w") as f:
        for word in vocab:
            f.write(word + "\n")


def parse_args():
    msg = "build vocabulary"
    parser = argparse.ArgumentParser(description=msg)

    msg = "input corpus"
    parser.add_argument("corpus", help=msg)
    msg = "output vocabulary name"
    parser.add_argument("output", default="vocab.txt", help=msg)
    msg = "limit"
    parser.add_argument("--limit", default=0, type=int, help=msg)
    msg = "add special token, separated by colon"
    parser.add_argument("--special", type=str, default="<pad>:<unk>",
                        help=msg)
    msg = "use lowercase"
    parser.add_argument("--lower", action="store_true", help=msg)

    return parser.parse_args()


def main(args):
    vocab = {}
    limit = args.limit
    count = 0

    words, labels, counts = count_items(args.corpus, args.lower)
    special = special_tokens(args.special)

    for token in special:
        vocab[token] = len(vocab)

    for word, freq in zip(words, counts):
        if limit and len(vocab) >= limit:
            break

        if word in vocab:
            print("warning: found duplicate token %s, ignored" % word)
            continue

        vocab[word] = len(vocab)
        count += freq

    save_vocab(args.output + "/vocab.txt", vocab)
    write_vocab(args.output + "/label.txt", labels)

    print("total words: %d" % sum(counts))
    print("unique words: %d" % len(words))
    print("vocabulary coverage: %4.2f%%" % (100.0 * count / sum(counts)))


if __name__ == "__main__":
    main(parse_args())
