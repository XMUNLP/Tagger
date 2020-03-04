# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import tagger.utils as utils

from tagger.modules.module import Module
from tagger.modules.affine import Affine
from tagger.modules.layer_norm import LayerNorm


class GRUCell(Module):

    def __init__(self, input_size, output_size, normalization=False,
                 name="gru"):
        super(GRUCell, self).__init__(name=name)

        self.input_size = input_size
        self.output_size = output_size

        with utils.scope(name):
            self.reset_gate = Affine(input_size + output_size, output_size,
                                     bias=False, name="reset_gate")
            self.update_gate = Affine(input_size + output_size, output_size,
                                      bias=False, name="update_gate")
            self.transform = Affine(input_size + output_size, output_size,
                                    name="transform")

    def forward(self, x, h):
        r = torch.sigmoid(self.reset_gate(torch.cat([x, h], -1)))
        u = torch.sigmoid(self.update_gate(torch.cat([x, h], -1)))
        c = self.transform(torch.cat([x, r * h], -1))

        new_h = (1.0 - u) * h + u * torch.tanh(h)

        return new_h, new_h

    def init_state(self, batch_size, dtype, device):
        h = torch.zeros([batch_size, self.output_size], dtype=dtype,
                        device=device)
        return h

    def mask_state(self, h, prev_h, mask):
        mask = mask[:, None]
        new_h = mask * h + (1.0 - mask) * prev_h
        return new_h

    def reset_parameters(self, initializer="uniform"):
        if initializer == "uniform_scaling":
            nn.init.xavier_uniform_(self.gates.weight)
            nn.init.constant_(self.gates.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.gates.weight, -0.08, 0.08)
            nn.init.uniform_(self.gates.bias, -0.08, 0.08)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class LSTMCell(Module):

    def __init__(self, input_size, output_size, normalization=False,
                 activation=torch.tanh, name="lstm"):
        super(LSTMCell, self).__init__(name=name)

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        with utils.scope(name):
            self.gates = Affine(input_size + output_size, 4 * output_size,
                                name="gates")
            if normalization:
                self.layer_norm = LayerNorm([4, output_size])
            else:
                self.layer_norm = None

        self.reset_parameters()

    def forward(self, x, state):
        c, h = state

        gates = self.gates(torch.cat([x, h], 1))

        if self.layer_norm is not None:
            combined = self.layer_norm(
                torch.reshape(gates, [-1, 4, self.output_size]))
        else:
            combined = torch.reshape(gates, [-1, 4, self.output_size])

        i, j, f, o = torch.unbind(combined, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)

        new_c = f * c + i * torch.tanh(j)

        if self.activation is None:
            # Do not use tanh activation
            new_h = o * new_c
        else:
            new_h = o * self.activation(new_c)

        return new_h, (new_c, new_h)

    def init_state(self, batch_size, dtype, device):
        c = torch.zeros([batch_size, self.output_size], dtype=dtype,
                        device=device)
        h = torch.zeros([batch_size, self.output_size], dtype=dtype,
                        device=device)
        return c, h

    def mask_state(self, state, prev_state, mask):
        c, h = state
        prev_c, prev_h = prev_state
        mask = mask[:, None]
        new_c = mask * c + (1.0 - mask) * prev_c
        new_h = mask * h + (1.0 - mask) * prev_h
        return new_c, new_h

    def reset_parameters(self, initializer="orthogonal"):
        if initializer == "uniform_scaling":
            nn.init.xavier_uniform_(self.gates.weight)
            nn.init.constant_(self.gates.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.gates.weight, -0.04, 0.04)
            nn.init.uniform_(self.gates.bias, -0.04, 0.04)
        elif initializer == "orthogonal":
            self.gates.orthogonal_initialize()
        else:
            raise ValueError("Unknown initializer %d" % initializer)



class HighwayLSTMCell(Module):

    def __init__(self, input_size, output_size, name="lstm"):
        super(HighwayLSTMCell, self).__init__(name=name)

        self.input_size = input_size
        self.output_size = output_size

        with utils.scope(name):
            self.gates = Affine(input_size + output_size, 5 * output_size,
                                name="gates")
            self.trans = Affine(input_size, output_size, name="trans")

        self.reset_parameters()

    def forward(self, x, state):
        c, h = state

        gates = self.gates(torch.cat([x, h], 1))
        combined = torch.reshape(gates, [-1, 5, self.output_size])
        i, j, f, o, t = torch.unbind(combined, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        t = torch.sigmoid(t)

        new_c = f * c + i * torch.tanh(j)
        tmp_h = o * torch.tanh(new_c)
        new_h = t * tmp_h + (1.0 - t) * self.trans(x)

        return new_h, (new_c, new_h)

    def init_state(self, batch_size, dtype, device):
        c = torch.zeros([batch_size, self.output_size], dtype=dtype,
                        device=device)
        h = torch.zeros([batch_size, self.output_size], dtype=dtype,
                        device=device)
        return c, h

    def mask_state(self, state, prev_state, mask):
        c, h = state
        prev_c, prev_h = prev_state
        mask = mask[:, None]
        new_c = mask * c + (1.0 - mask) * prev_c
        new_h = mask * h + (1.0 - mask) * prev_h
        return new_c, new_h

    def reset_parameters(self, initializer="orthogonal"):
        if initializer == "uniform_scaling":
            nn.init.xavier_uniform_(self.gates.weight)
            nn.init.constant_(self.gates.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.gates.weight, -0.04, 0.04)
            nn.init.uniform_(self.gates.bias, -0.04, 0.04)
        elif initializer == "orthogonal":
            self.gates.orthogonal_initialize()
            self.trans.orthogonal_initialize()
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class DynamicLSTMCell(Module):

    def __init__(self, input_size, output_size, k=2, num_cells=4, name="lstm"):
        super(DynamicLSTMCell, self).__init__(name=name)

        self.input_size = input_size
        self.output_size = output_size
        self.num_cells = num_cells
        self.k = k

        with utils.scope(name):
            self.gates = Affine(input_size + output_size,
                                4 * output_size * num_cells,
                                name="gates")
            self.topk_gate = Affine(input_size + output_size,
                                    num_cells, name="controller")


        self.reset_parameters()

    @staticmethod
    def top_k_softmax(logits, k, n):
        top_logits, top_indices = torch.topk(logits, k=min(k + 1, n))

        top_k_logits = top_logits[:, :k]
        top_k_indices = top_indices[:, :k]

        probs = torch.softmax(top_k_logits, dim=-1)
        batch = top_k_logits.shape[0]
        k = top_k_logits.shape[1]

        # Flat to 1D
        indices_flat = torch.reshape(top_k_indices, [-1])
        indices_flat = indices_flat + torch.div(
            torch.arange(batch * k, device=logits.device), k) * n

        tensor = torch.zeros([batch * n], dtype=logits.dtype,
                             device=logits.device)
        tensor = tensor.scatter_add(0, indices_flat.long(),
                                    torch.reshape(probs, [-1]))

        return torch.reshape(tensor, [batch, n])

    def forward(self, x, state):
        c, h = state
        feats = torch.cat([x, h], dim=-1)

        logits = self.topk_gate(feats)
        # [batch, num_cells]
        gate = self.top_k_softmax(logits, self.k, self.num_cells)

        # [batch, 4 * num_cells * dim]
        combined = self.gates(feats)
        combined = torch.reshape(combined,
                                 [-1, self.num_cells, 4, self.output_size])

        i, j, f, o = torch.unbind(combined, 2)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)

        # [batch, num_cells, dim]
        new_c = f * c[:, None, :] + i * torch.tanh(j)
        new_h = o * torch.tanh(new_c)

        gate = gate[:, None, :]
        new_c = torch.matmul(gate, new_c)
        new_h = torch.matmul(gate, new_h)

        new_c = torch.squeeze(new_c, 1)
        new_h = torch.squeeze(new_h, 1)

        return new_h, (new_c, new_h)

    def init_state(self, batch_size, dtype, device):
        c = torch.zeros([batch_size, self.output_size], dtype=dtype,
                        device=device)
        h = torch.zeros([batch_size, self.output_size], dtype=dtype,
                        device=device)
        return c, h

    def mask_state(self, state, prev_state, mask):
        c, h = state
        prev_c, prev_h = prev_state
        mask = mask[:, None]
        new_c = mask * c + (1.0 - mask) * prev_c
        new_h = mask * h + (1.0 - mask) * prev_h
        return new_c, new_h

    def reset_parameters(self, initializer="orthogonal"):
        if initializer == "uniform_scaling":
            nn.init.xavier_uniform_(self.gates.weight)
            nn.init.constant_(self.gates.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.gates.weight, -0.04, 0.04)
            nn.init.uniform_(self.gates.bias, -0.04, 0.04)
        elif initializer == "orthogonal":
            weight = self.gates.weight.view(
                [self.input_size + self.output_size, self.num_cells,
                 4 * self.output_size])
            nn.init.orthogonal_(weight, 1.0)
            nn.init.constant_(self.gates.bias, 0.0)
        else:
            raise ValueError("Unknown initializer %d" % initializer)
