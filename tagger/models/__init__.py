# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tagger.models.deepatt


def get_model(name):
    name = name.lower()

    if name == "deepatt":
        return tagger.models.deepatt.DeepAtt
    else:
        raise LookupError("Unknown model %s" % name)
