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

from typing import List, Optional, Tuple, Type

import numpy as np
from mxnet import gluon

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.block.decoder import Seq2SeqDecoder
from gluonts.mx.block.enc2dec import Seq2SeqEnc2Dec
from gluonts.mx.block.encoder import Seq2SeqEncoder
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.quantile_output import QuantileOutput
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.util import weighted_average


class ForkingSeq2SeqNetworkBase(gluon.HybridBlock):
    """
    Base network for the :class:`ForkingSeq2SeqEstimator`.

    Parameters
    ----------
    encoder: Seq2SeqEncoder
        encoder block.
    enc2dec: Seq2SeqEnc2Dec
        encoder to decoder mapping block.
    decoder: Seq2SeqDecoder
        decoder block.
    quantile_output
        quantile output
    distr_output
        distribution output
    context_length: int,
        length of the encoding sequence.
    cardinality: List[int],
        number of values of each categorical feature.
    embedding_dimension: List[int],
        dimension of the embeddings for categorical features.
    scaling
        Whether to automatically scale the target values. (default: True)
    scaling_decoder_dynamic_feature
        Whether to automatically scale the dynamic features for the decoder.
        (default: False)
    dtype
        (default: np.float32)
    num_forking: int,
        decides how much forking to do in the decoder. 1 reduces to seq2seq and
        enc_len reduces to MQ-C(R)NN.
    kwargs: dict
        dictionary of Gluon HybridBlock parameters
    """

    @validated()
    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        enc2dec: Seq2SeqEnc2Dec,
        decoder: Seq2SeqDecoder,
        context_length: int,
        cardinality: List[int],
        embedding_dimension: List[int],
        distr_output: Optional[DistributionOutput] = None,
        quantile_output: Optional[QuantileOutput] = None,
        scaling: bool = True,
        scaling_decoder_dynamic_feature: bool = False,
        dtype: Type = np.float32,
        num_forking: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert (distr_output is None) != (quantile_output is None)

        self.encoder = encoder
        self.enc2dec = enc2dec
        self.decoder = decoder
        self.distr_output = distr_output
        self.quantile_output = quantile_output
        self.scaling = scaling
        self.scaling_decoder_dynamic_feature = scaling_decoder_dynamic_feature
        self.dtype = dtype
        self.num_forking = (
            num_forking if num_forking is not None else context_length
        )

        if self.scaling:
            self.scaler = MeanScaler()
        else:
            self.scaler = NOPScaler()

        if self.scaling_decoder_dynamic_feature:
            self.scaler_decoder_dynamic_feature = MeanScaler(axis=1)
        else:
            self.scaler_decoder_dynamic_feature = NOPScaler(axis=1)

        with self.name_scope():
            if self.quantile_output:
                self.quantile_proj = self.quantile_output.get_quantile_proj()
                self.loss = self.quantile_output.get_loss()
            else:
                assert self.distr_output is not None
                self.distr_args_proj = self.distr_output.get_args_proj()
            self.embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=embedding_dimension,
                dtype=self.dtype,
            )

    # this method connects the sub-networks and returns the decoder output
    def get_decoder_network_output(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic: Tensor,
        future_feat_dynamic: Tensor,
        feat_static_cat: Tensor,
        past_observed_values: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: Tensor
            shape (batch_size, encoder_length, 1)
        past_feat_dynamic
            shape (batch_size, encoder_length, num_past_feat_dynamic)
        future_feat_dynamic
            shape (batch_size, num_forking, decoder_length, num_feat_dynamic)
        feat_static_cat
            shape (batch_size, num_feat_static_cat)
        past_observed_values: Tensor
            shape (batch_size, encoder_length, 1)
        Returns
        -------
        decoder output tensor of size (batch_size, num_forking, dec_len,
        decoder_mlp_dim_seq[0])
        """
        pass


class ForkingSeq2SeqTrainingNetwork(ForkingSeq2SeqNetworkBase):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        future_target: Tensor,
        past_feat_dynamic: Tensor,
        future_feat_dynamic: Tensor,
        feat_static_cat: Tensor,
        past_observed_values: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: Tensor
            shape (batch_size, encoder_length, 1)
        future_target: Tensor
            shape (batch_size, num_forking, decoder_length)
        past_feat_dynamic
            shape (batch_size, encoder_length, num_past_feat_dynamic)
        future_feat_dynamic
            shape (batch_size, num_forking, decoder_length, num_feat_dynamic)
        feat_static_cat
            shape (batch_size, num_feat_static_cat)
        past_observed_values: Tensor
            shape (batch_size, encoder_length, 1)
        future_observed_values: Tensor
            shape (batch_size, num_forking, decoder_length)

        Returns
        -------
        loss with shape (batch_size, prediction_length)
        """
        pass


class ForkingSeq2SeqPredictionNetwork(ForkingSeq2SeqNetworkBase):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic: Tensor,
        future_feat_dynamic: Tensor,
        feat_static_cat: Tensor,
        past_observed_values: Tensor,
    ) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor]:
        """
        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: Tensor
             shape (batch_size, encoder_length, 1)
        feat_static_cat
            shape (batch_size, num_feat_static_cat)
        past_feat_dynamic
            shape (batch_size, encoder_length, num_past_feat_dynamic)
        future_feat_dynamic
            shape (batch_size, num_forking, decoder_length, num_feat_dynamic)
        past_observed_values: Tensor
            shape (batch_size, encoder_length, 1)

        Returns
        -------
        prediction tensor with shape (num_test_ts, num_quantiles,
        prediction_length)
        """
        pass


class ForkingSeq2SeqDistributionPredictionNetwork(ForkingSeq2SeqNetworkBase):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic: Tensor,
        future_feat_dynamic: Tensor,
        feat_static_cat: Tensor,
        past_observed_values: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        F: mx.symbol or mx.ndarray
            Gluon function space
        past_target: Tensor
             shape (batch_size, encoder_length, 1)
        feat_static_cat
            shape (batch_size, encoder_length, num_feature_static_cat)
        past_feat_dynamic
            shape (batch_size, encoder_length, num_feature_dynamic)
        future_feat_dynamic
            shape (batch_size, num_forking, decoder_length,
            num_feature_dynamic)
        past_observed_values: Tensor
            shape (batch_size, encoder_length, 1)
        Returns
        -------
        distr_args: the parameters of distribution
        loc: an array of zeros with the same shape of scale
        scale:
        """
        pass
