# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Dict, Optional, Tuple

import mxnet as mx
from mxnet.gluon import HybridBlock

from gluonts.mx import Tensor


def split_heads(F, x: Tensor, dim_per_head: int, heads: int) -> Tensor:
    r"""
    Returns a tensor with head dimension folded into batch and last dimension
    divided by the number of heads.

    Parameters
    ----------
    x
        Tensor of shape (batch_size, time_length, dim).
    dim_per_head
        Dimension per head
    heads
        Number of heads

    Returns
    -------
    Tensor of shape (batch_size * heads, time_length, dim_per_head).
    """
    pass


def dot_attention(
    F,
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Optional[Tensor] = None,
    dropout: float = 0.0,
) -> Tensor:
    r"""

    Parameters
    ----------
    queries
        Attention queries of shape (n, lq, d)
    keys
        Attention keys of shape (n, lk, d)
    values
        Attention values of shape (n, lk, dv)
    mask
        Optional mask tensor
    dropout
        Dropout rate

    Returns
    -------
    'Context' vectors for each query of shape (n, lq, dv)
    """
    pass


def combine_heads(F, x: Tensor, dim_per_head: int, heads: int) -> Tensor:
    r"""

    Parameters
    ----------
    x
        Tensor of shape (batch_size * heads, time_length, dim_per_head)
    dim_per_head
        Dimension per head
    heads
        Number of heads

    Returns
    -------
    Tensor of shape (batch_size, time_length, dim)
    """
    pass


class LayerNormalization(HybridBlock):
    """
    Implements layer normalization as proposed in [BKH16]_.
    """

    def __init__(
        self,
        scale_init: str = "ones",
        shift_init: str = "zeros",
        eps: float = 1e-06,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.scale_init = scale_init
        self.shift_init = shift_init

        with self.name_scope():
            self.lnorm = mx.gluon.nn.LayerNorm(
                axis=-1,
                gamma_initializer=self.scale_init,
                beta_initializer=self.shift_init,
                epsilon=eps,
            )

    def hybrid_forward(self, F, data: Tensor) -> Tensor:
        r"""
        Normalizes hidden units of data as follows:

        data = scale * (data - mean) / sqrt(var + eps) + shift

        Normalization is performed over the last dimension of the input data.

        Parameters
        ----------
        data
            Data to normalize of shape (d0, ..., dn, num_hidden)

        Returns
        -------
        Normalized inputs of shape: (d0, ..., dn, num_hidden)
        """
        pass


class InputLayer(HybridBlock):
    r"""
    Transforms the input vector to model_size with an one-layer MPL, i.e.,
    (batch_size, time_length, input_dim) -> (batch_size, time_length,
    model_size)
    """

    def __init__(self, model_size: int = 64, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model_size = model_size
        with self.name_scope():
            self.net = mx.gluon.nn.Dense(units=self.model_size, flatten=False)



class MultiHeadAttentionBase(HybridBlock):
    """
    Base class for Multi-head attention.

    Parameters
    ----------
    att_dim_in
        Attention dimension (number of hidden units)
    heads
        Number of attention heads
    att_dim_out
        Output dimension (number of output units)
    dropout
        Dropout rate on attention scores
    """

    def __init__(
        self,
        att_dim_in: int = 32,
        heads: int = 8,
        att_dim_out: int = 32,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert (
            att_dim_in % heads == 0
        ), "Number of heads {} must divide attention att_dim_in {}".format(
            heads, att_dim_in
        )

        self.att_dim_in = att_dim_in
        self.heads = heads
        self.att_dim_out = att_dim_out
        self.dropout = dropout
        self.dim_per_head = self.att_dim_in // self.heads

        with self.name_scope():
            self.dense_att = mx.gluon.nn.Dense(
                units=self.att_dim_out, flatten=False
            )

    def _attend(
        self,
        F,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Returns context vectors of multi-head dot attention.

        Parameters
        ----------
        queries
            Queries tensor of shape (batch_size, query_max_length, dim)
        keys
            Keys tensor of shape (batch_size, memory_max_length, dim)
        values
            Values tensor of shape (batch_size, memory_max_length, dim)
        mask

        Returns
        -------
        Context vectors of shape (batch_size, query_max_length, att_dim_out)
        """
        pass


class MultiHeadSelfAttention(MultiHeadAttentionBase):
    r"""
    Multi-head self-attention. Independent linear projections of inputs serve
    as queries, keys, and values for the attention.

    Parameters
    ----------
    att_dim_in
        Attention dimension (number of hidden units)
    heads
        Number of attention heads
    att_dim_out
        Output dimension (number of output units)
    dropout
        Dropout rate on attention scores
    """

    def __init__(
        self,
        att_dim_in: int = 32,
        heads: int = 8,
        att_dim_out: int = 32,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(att_dim_in, heads, att_dim_out, dropout, **kwargs)

        with self.name_scope():
            self.dense_pre_satt = mx.gluon.nn.Dense(
                units=self.att_dim_in * 3, flatten=False
            )

    def hybrid_forward(
        self,
        F,
        inputs: Tensor,
        mask: Optional[Tensor] = None,
        cache: Optional[Dict[str, Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, Optional[Dict]]:
        r"""
        Computes multi-head attention on a set of inputs, serving as queries,
        keys, and values. If sequence lengths are provided, they will be used
        to mask the attention scores. May also use a cache of previously
        computed inputs.

        Parameters
        ----------
        inputs
            Input data of shape (batch_size, max_length, att_dim_in)
        mask
            Optional tensor to mask attention scores
        cache
            Optional dictionary of previously computed keys and values

        Returns
        -------
        Tensor
            A tensor of shape (batch_size, max_length, att_dim_out)
        """
        pass


class MultiHeadAttention(MultiHeadAttentionBase):
    r"""
    Multi-head attention layer for queries independent from keys/values.

    Parameters
    ----------
    att_dim_in
        Attention dimension (number of hidden units)
    heads
        Number of attention heads
    att_dim_out
        Output dimension (number of output units)
    dropout
        Dropout rate on attention scores
    """

    def __init__(
        self,
        att_dim_in: int = 32,
        heads: int = 8,
        att_dim_out: int = 32,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(att_dim_in, heads, att_dim_out, dropout, **kwargs)

        with self.name_scope():
            self.dense_pre_att_q = mx.gluon.nn.Dense(
                units=self.att_dim_in, flatten=False
            )
            self.dense_pre_att_k = mx.gluon.nn.Dense(
                units=self.att_dim_in, flatten=False
            )
            self.dense_pre_att_v = mx.gluon.nn.Dense(
                units=self.att_dim_in, flatten=False
            )

    def hybrid_forward(
        self, F, queries: Tensor, memory: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Computes multi-head attention for queries given a memory tensor. If
        sequence lengths are provided, they will be used to mask the attention
        scores. A mask tensor may also be used to mask the attention scores.
        Returns a tensor of shape (batch_size, max_length, att_dim_out).

        Parameters
        ----------
        queries
            Queries tensor of shape (batch_size, query_max_length, att_dim_in)
        memory
            Memory tensor to attend to of shape (batch_size, memory_max_length,
            att_dim_in)
        mask
            Optional tensor to mask attention scores

        Returns
        -------
        Tensor of shape (batch_size, query_seq_len, att_dim_out)
        """
        pass


class TransformerFeedForward(HybridBlock):
    r"""
    Position-wise feed-forward network with activation.

    .. math::

        activation(XW_1 + b_1)W_2 + b_2

    :math:`W_1`: (batch_size, d, inner_dim)
    :math:`W_2`: (batch_size, inner_dim, out_dim)
    """

    def __init__(
        self,
        inner_dim: int = 32,  # W1: (batch_size, d, inner_dim)
        out_dim: int = 32,  # W2: (batch_size, inner_dim, out_dim)
        act_type: str = "softrelu",
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.inner_dim = inner_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act_type = act_type

        with self.name_scope():
            self.mlp = mx.gluon.nn.HybridSequential()
            self.mlp.add(
                mx.gluon.nn.Dense(
                    units=self.inner_dim,
                    use_bias=True,
                    activation=self.act_type,
                    flatten=False,
                )
            )
            if self.dropout > 0.0:
                self.mlp.add(mx.gluon.nn.Dropout(self.dropout))
            self.mlp.add(
                mx.gluon.nn.Dense(units=out_dim, use_bias=True, flatten=False)
            )  # no activation

    def hybrid_forward(self, F, x: Tensor, *args) -> Tensor:
        r"""
        Position-wise feed-forward network with activation.

        Parameters
        ----------
        x
            Tensor of shape (batch_size, d, in_dim)

        Returns
        -------
        Tensor of shape (batch_size, d1, out_dim)
        """
        pass


class TransformerProcessBlock(HybridBlock):
    r"""
    Block to perform pre/post processing on layer inputs.

    The processing steps are determined by the sequence argument, which can
    contain one of the three operations:
    n: layer normalization
    r: residual connection
    d: dropout
    """

    def __init__(self, sequence: str, dropout: float, **kwargs) -> None:
        super().__init__(**kwargs)

        self.sequence = sequence
        self.dropout = dropout
        self.layer_norm = None
        if "n" in sequence:
            self.layer_norm = LayerNormalization()

    def hybrid_forward(
        self, F, data: Tensor, prev: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Apply processing sequence to data with optional previous input.

        Parameters
        ----------
        data
            Input data of shape: (batch_size, length, num_hidden)
        prev
            Previous data of shape (batch_size, length, num_hidden)

        Returns
        -------
        Processed data of shape (batch_size, length, num_hidden).
        """
        pass
