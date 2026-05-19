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

from typing import Optional, Type

import mxnet as mx
from mxnet.gluon import loss, nn

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.block.scaler import MeanScaler, NOPScaler


class LSTNetBase(nn.HybridBlock):
    @validated()
    def __init__(
        self,
        num_series: int,
        channels: int,
        kernel_size: int,
        rnn_cell_type: str,
        rnn_num_layers: int,
        rnn_num_cells: int,
        skip_rnn_cell_type: str,
        skip_rnn_num_layers: int,
        skip_rnn_num_cells: int,
        skip_size: int,
        ar_window: int,
        context_length: int,
        lead_time: int,
        prediction_length: int,
        dropout_rate: float,
        output_activation: Optional[str],
        scaling: bool,
        dtype: Type,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_series = num_series
        self.channels = channels
        assert (
            channels % skip_size == 0
        ), "number of conv2d `channels` must be divisible by the `skip_size`"
        self.skip_size = skip_size
        assert (
            ar_window > 0
        ), "auto-regressive window must be a positive integer"
        self.ar_window = ar_window
        assert lead_time >= 0, "`lead_time` must be greater than zero"
        assert (
            prediction_length > 0
        ), "`prediction_length` must be greater than zero"
        self.prediction_length = prediction_length
        self.horizon = lead_time
        assert context_length > 0, "`context_length` must be greater than zero"
        self.context_length = context_length
        if output_activation is not None:
            assert output_activation in [
                "sigmoid",
                "tanh",
            ], "`output_activation` must be either 'sigmiod' or 'tanh' "
        self.output_activation = output_activation
        assert rnn_cell_type in [
            "gru",
            "lstm",
        ], "`rnn_cell_type` must be either 'gru' or 'lstm' "
        assert skip_rnn_cell_type in [
            "gru",
            "lstm",
        ], "`skip_rnn_cell_type` must be either 'gru' or 'lstm' "
        self.conv_out = context_length - kernel_size + 1
        self.conv_skip = self.conv_out // skip_size
        assert self.conv_skip > 0, (
            "conv2d output size must be greater than or equal to `skip_size`\n"
            "Choose a smaller `kernel_size` or bigger `context_length`"
        )
        self.channel_skip_count = self.conv_skip * skip_size
        self.dtype = dtype
        with self.name_scope():
            self.cnn = nn.Conv2D(
                channels,
                (num_series, kernel_size),
                activation="relu",
                layout="NCHW",
                in_channels=2,
            )  # NC1T
            self.cnn.cast(dtype)
            self.dropout = nn.Dropout(dropout_rate)
            self.rnn = self._create_rnn_layer(
                rnn_num_cells, rnn_num_layers, rnn_cell_type, dropout_rate
            )  # NTC
            self.rnn.cast(dtype)
            self.skip_rnn_num_cells = skip_rnn_num_cells
            self.skip_rnn = self._create_rnn_layer(
                skip_rnn_num_cells,
                skip_rnn_num_layers,
                skip_rnn_cell_type,
                dropout_rate,
            )  # NTC
            self.skip_rnn.cast(dtype)
            # TODO: add temporal attention option
            self.fc = nn.Dense(num_series, dtype=dtype)
            self.ar_fc = nn.Dense(
                prediction_length, dtype=dtype, flatten=False
            )
            if scaling:
                self.scaler = MeanScaler(axis=2, keepdims=True)
            else:
                self.scaler = NOPScaler(axis=2, keepdims=True)

    @staticmethod
    def _create_rnn_layer(
        num_cells: int, num_layers: int, cell_type: str, dropout_rate: float
    ) -> nn.HybridBlock:
        # TODO: GRUCell activation is fixed to tanh
        RnnCell = {"lstm": mx.gluon.rnn.LSTMCell, "gru": mx.gluon.rnn.GRUCell}[
            cell_type
        ]
        rnn = mx.gluon.rnn.HybridSequentialRNNCell()
        for _ in range(num_layers):
            cell = RnnCell(hidden_size=num_cells)
            cell = (
                mx.gluon.rnn.ZoneoutCell(cell, zoneout_states=dropout_rate)
                if dropout_rate > 0.0
                else cell
            )
            rnn.add(cell)
        return rnn



    def hybrid_forward(
        self, F, past_target: Tensor, past_observed_values: Tensor
    ) -> Tensor:
        """
        Given the tensor `past_target`, first we normalize it by the
        `past_observed_values` which is an indicator tensor with 0 or 1 values.
        Then it outputs the result of LSTNet.

        Parameters
        ----------
        F
        past_target
            Tensor of shape (batch_size, num_series, context_length)
        past_observed_values
            Tensor of shape (batch_size, num_series, context_length)

        Returns
        -------
        Tensor
            Shape (batch_size, num_series, 1) if `horizon` was specified
            and of shape (batch_size, num_series, prediction_length)
            if `prediction_length` was provided
        """
        pass


class LSTNetTrain(LSTNetBase):
    @validated()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = loss.L1Loss()

    def hybrid_forward(  # type: ignore
        self,
        F,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        """
        Computes the training l1 loss for LSTNet for multivariate time series.
        All input tensors have NCT layout.

        Parameters
        ----------
        F
        past_target
            Tensor of shape (batch_size, num_series, context_length)
        past_observed_values
            Tensor of shape (batch_size, num_series, context_length)
        future_target
            Tensor of shape (batch_size, num_series, prediction_length)
        future_observed_values
            Tensor of shape (batch_size, num_series, prediction_length)

        Returns
        -------
        Tensor
            Loss values of shape (batch_size,)
        """
        pass


class LSTNetPredict(LSTNetBase):
    def hybrid_forward(
        self, F, past_target: Tensor, past_observed_values: Tensor
    ) -> Tensor:
        """
        Returns the LSTNet predictions. All tensors have NCT layout.

        Parameters
        ----------
        F
        past_target
            Tensor of shape (batch_size, num_series, context_length)
        past_observed_values
            Tensor of shape (batch_size, num_series, context_length)

        Returns
        -------
        Tensor
            Predicted samples of shape (batch_size, num_samples,
            prediction_length, num_series)
        """
        pass
