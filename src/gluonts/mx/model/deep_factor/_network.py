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

from typing import Tuple

import math

from mxnet.gluon import HybridBlock, nn

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.block.feature import FeatureEmbedder


class DeepFactorNetworkBase(HybridBlock):
    def __init__(
        self,
        global_model: HybridBlock,
        local_model: HybridBlock,
        embedder: FeatureEmbedder,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.global_model = global_model
        self.local_model = local_model
        self.embedder = embedder
        with self.name_scope():
            self.loading = nn.Dense(
                units=global_model.num_output, use_bias=False
            )





class DeepFactorTrainingNetwork(DeepFactorNetworkBase):
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, 1)
        past_time_feat: Tensor,
        # (batch_size, history_length, num_features)
        past_target: Tensor,  # (batch_size, history_length)
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            Function space
        feat_static_cat
            Shape: (batch_size, 1)
        past_time_feat
            Shape: (batch_size, history_length, num_features)
        past_target
            Shape: (batch_size, history_length)

        Returns
        -------
        Tensor
            A batch of negative log likelihoods.
        """
        pass


class DeepFactorPredictionNetwork(DeepFactorNetworkBase):
    @validated()
    def __init__(
        self, prediction_len: int, num_parallel_samples: int, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.prediction_len = prediction_len
        self.num_parallel_samples = num_parallel_samples

    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        past_target: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            Function space
        feat_static_cat
            Shape: (batch_size, 1)
        past_time_feat
            Shape: (batch_size, history_length, num_features)
        future_time_feat
            Shape: (batch_size, prediction_length, num_features)
        past_target
            Shape: (batch_size, history_length)

        Returns
        -------
        Tensor
            Samples of shape (batch_size, num_samples, prediction_length).
        """
        pass
