# py_func.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy as np


def tensor_to_string(input_tensor, vocab_tensor):
    """ Convert integer tensor to string tensor
        Args:
            input_tensor: shape [batch, max_length]
            vocab_tensor: shape [max_vocab]
        Returns:
            a Tensor with shape [batch], type tf.string
    """
    str_list = []

    for seq in input_tensor:
        str_seq = []
        for word_id in seq:
            if word_id == 0 or word_id == 1:
                break
            str_seq.append(vocab_tensor[word_id])
        str_list.append(" ".join(str_seq))

    return np.array(str_list)
