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

from typing import List, Optional, Tuple

import mxnet as mx
import numpy as np
from mxnet import nd

from gluonts.core.component import validated
from gluonts.mx.model.tpp import distribution
from gluonts.mx.model.tpp.distribution.base import TPPDistributionOutput
from gluonts.mx import Tensor
from gluonts.mx.distribution import CategoricalOutput


class DeepTPPNetworkBase(mx.gluon.HybridBlock):
    """
    Temporal point process model based on a recurrent neural network.

    Parameters
    ----------
    num_marks
        Number of discrete marks (correlated processes), that are available
        in the data.
    interval_length
        The length of the total time interval that is in the prediction
        range. Note that in contrast to discrete-time models in the rest
        of GluonTS, the network is trained to predict an interval, in
        continuous time.
    time_distr_output
        Output distribution for the inter-arrival times. Available
        distributions can be found in gluonts.model.tpp.distribution.
    embedding_dim
        Dimension of vector embeddings of marks (used only as input).
    num_hidden_dimensions
        Number of hidden units in the RNN.
    output_scale
        Positive scaling applied to the inter-event times. You should provide
        this argument if the average inter-arrival time is much larger than 1.
    apply_log_to_rnn_inputs
        Apply logarithm to inter-event times that are fed into the RNN.
    """

    @validated()
    def __init__(
        self,
        num_marks: int,
        interval_length: float,
        time_distr_output: TPPDistributionOutput = distribution.WeibullOutput(),  # noqa: E501
        embedding_dim: int = 5,
        num_hidden_dimensions: int = 10,
        output_scale: Optional[Tensor] = None,
        apply_log_to_rnn_inputs: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_marks = num_marks
        self.interval_length = interval_length
        self.rnn_hidden_size = num_hidden_dimensions
        self.output_scale = output_scale
        self.apply_log_to_rnn_inputs = apply_log_to_rnn_inputs

        with self.name_scope():
            self.embedding = mx.gluon.nn.Embedding(
                input_dim=num_marks, output_dim=embedding_dim
            )
            self.rnn = mx.gluon.rnn.GRU(
                num_hidden_dimensions,
                input_size=embedding_dim + 1,
                layout="NTC",
            )
            # Conditional distribution over the inter-arrival times
            self.time_distr_output = time_distr_output
            self.time_distr_args_proj = self.time_distr_output.get_args_proj()
            # Conditional distribution over the marks
            if num_marks > 1:
                self.mark_distr_output = CategoricalOutput(num_marks)
                self.mark_distr_args_proj = (
                    self.mark_distr_output.get_args_proj()
                )

    def hybridize(self, active=True, **kwargs):
        if active:
            raise NotImplementedError(
                "DeepTPP blocks do not support hybridization"
            )


class DeepTPPTrainingNetwork(DeepTPPNetworkBase):
    def hybrid_forward(
        self,
        F,
        target: Tensor,
        valid_length: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Computes the negative log likelihood loss for the given sequences.

        As the model is trained on past (resp. future) or context
        (resp. prediction) "intervals" as opposed to fixed-length "sequences",
        the number of data points available varies across observations. To
        account for this, data is made available to the training network as a
        "ragged" tensor. The number of valid entries in each sequence is
        provided in a separate variable, :code:`xxx_valid_length`.

        Parameters
        ----------
        F
            MXNet backend.
        target
            Tensor with observations.
            Shape: (batch_size, past_max_sequence_length, target_dim).
        valid_length
            The `valid_length` or number of valid entries in the past_target
            Tensor. Shape: (batch_size,)

        Returns
        -------
        Tensor
            Loss tensor. Shape: (batch_size,).
        """
        pass


class DeepTPPPredictionNetwork(DeepTPPNetworkBase):
    @validated()
    def __init__(
        self,
        prediction_interval_length: float,
        num_parallel_samples: int = 100,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.prediction_interval_length = prediction_interval_length

    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_valid_length: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Draw forward samples from the model. At each step, we sample an inter-
        event time and feed it into the RNN to obtain the parameters for the
        next distribution over the inter-event time.

        Parameters
        ----------
        F
            MXNet backend.
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim). Has to comply
            with :code:`self.context_interval_length`.
        past_valid_length
            The `valid_length` or number of valid entries in the past_target
            Tensor. Shape: (batch_size,)

        Returns
        -------
        sampled_target: Tensor
            Predicted inter-event times and marks.
            Shape: (samples, batch_size, max_prediction_length, target_dim).
        sampled_valid_length: Tensor
            The number of valid entries in the time axis of each sample.
            Shape (samples, batch_size)
        """
        pass
