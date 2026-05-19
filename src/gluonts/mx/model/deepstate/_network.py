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

from typing import List, Optional

import mxnet as mx

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.distribution.lds import LDS, LDSArgsProj, ParameterBounds
from gluonts.mx.util import make_nd_diag, weighted_average

from .issm import ISSM


class DeepStateNetwork(mx.gluon.HybridBlock):
    @validated()
    def __init__(
        self,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        past_length: int,
        prediction_length: int,
        issm: ISSM,
        dropout_rate: float,
        cardinality: List[int],
        embedding_dimension: List[int],
        scaling: bool = True,
        noise_std_bounds: ParameterBounds = ParameterBounds(1e-6, 1.0),
        prior_cov_bounds: ParameterBounds = ParameterBounds(1e-6, 1.0),
        innovation_bounds: ParameterBounds = ParameterBounds(1e-6, 0.01),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.past_length = past_length
        self.prediction_length = prediction_length
        self.issm = issm
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_cat = len(cardinality)
        self.scaling = scaling

        assert len(cardinality) == len(embedding_dimension), (
            "embedding_dimension should be a list with the same size as"
            " cardinality"
        )
        self.univariate = self.issm.output_dim() == 1

        self.noise_std_bounds = noise_std_bounds
        self.prior_cov_bounds = prior_cov_bounds
        self.innovation_bounds = innovation_bounds

        with self.name_scope():
            self.prior_mean_model = mx.gluon.nn.Dense(
                units=self.issm.latent_dim(), flatten=False
            )
            self.prior_cov_diag_model = mx.gluon.nn.Dense(
                units=self.issm.latent_dim(),
                activation="sigmoid",
                flatten=False,
            )
            self.lstm = mx.gluon.rnn.HybridSequentialRNNCell()
            self.lds_proj = LDSArgsProj(
                output_dim=self.issm.output_dim(),
                noise_std_bounds=self.noise_std_bounds,
                innovation_bounds=self.innovation_bounds,
            )
            for k in range(num_layers):
                cell = mx.gluon.rnn.LSTMCell(hidden_size=num_cells)
                cell = mx.gluon.rnn.ResidualCell(cell) if k > 0 else cell
                cell = (
                    mx.gluon.rnn.ZoneoutCell(cell, zoneout_states=dropout_rate)
                    if dropout_rate > 0.0
                    else cell
                )
                self.lstm.add(cell)
            self.embedder = FeatureEmbedder(
                cardinalities=cardinality, embedding_dims=embedding_dimension
            )
            if scaling:
                self.scaler = MeanScaler(keepdims=False)
            else:
                self.scaler = NOPScaler(keepdims=False)



class DeepStateTrainingNetwork(DeepStateNetwork):


class DeepStatePredictionNetwork(DeepStateNetwork):
    @validated()
    def __init__(self, num_parallel_samples: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

