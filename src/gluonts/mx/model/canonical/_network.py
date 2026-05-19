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

from mxnet.gluon import HybridBlock

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.scaler import MeanScaler
from gluonts.mx.distribution import DistributionOutput


class CanonicalNetworkBase(HybridBlock):
    @validated()
    def __init__(
        self,
        model: HybridBlock,
        embedder: FeatureEmbedder,
        distr_output: DistributionOutput,
        is_sequential: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.distr_output = distr_output
        self.is_sequential = is_sequential
        self.model = model
        self.embedder = embedder

        with self.name_scope():
            self.proj_distr_args = self.distr_output.get_args_proj()
            self.scaler = MeanScaler(keepdims=True)


    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError


class CanonicalTrainingNetwork(CanonicalNetworkBase):
    def hybrid_forward(  # type: ignore
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,
        # (batch_size, num_features, history_length)
        past_target: Tensor,  # (batch_size, history_length)
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            Function space
        feat_static_cat
            Shape: (batch_size, num_features)
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


class CanonicalPredictionNetwork(CanonicalNetworkBase):
    @validated()
    def __init__(
        self, prediction_len: int, num_parallel_samples: int, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.prediction_len = prediction_len
        self.num_parallel_samples = num_parallel_samples

    def hybrid_forward(  # type: ignore
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
            Function space module.
        feat_static_cat
            Shape: (batch_size, num_features).
        past_time_feat
            Shape: (batch_size, history_length, num_features).
        future_time_feat
            Shape: (batch_size, history_length, num_features).
        past_target
            Shape: (batch_size, history_length).

        Returns
        -------
        Tensor
            a batch of prediction samples
            Shape: (batch_size, prediction_length, num_sample_paths)
        """
        pass
