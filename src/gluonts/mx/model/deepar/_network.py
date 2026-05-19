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

from typing import List, Optional, Tuple, Union, Type

import mxnet as mx
import numpy as np
from mxnet.gluon.contrib.rnn import VariationalDropoutCell
from mxnet.gluon.rnn import ZoneoutCell

from gluonts.core.component import validated
from gluonts.itertools import prod
from gluonts.mx import Tensor
from gluonts.mx.block.dropout import RNNZoneoutCell, VariationalZoneoutCell
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.regularization import (
    ActivationRegularizationLoss,
    TemporalActivationRegularizationLoss,
)
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.distribution import Distribution, DistributionOutput
from gluonts.mx.distribution.distribution import getF
from gluonts.mx.util import weighted_average, mx_switch
from mxnet import autograd


class DeepARNetwork(mx.gluon.HybridBlock):
    @validated()
    def __init__(
        self,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        dropout_rate: float,
        cardinality: List[int],
        embedding_dimension: List[int],
        lags_seq: List[int],
        dropoutcell_type: str = "ZoneoutCell",
        scaling: bool = True,
        dtype: Type = np.float32,
        num_imputation_samples: int = 1,
        minimum_scale: float = 1e-10,
        impute_missing_values: bool = False,
        default_scale: Optional[float] = None,
        nonnegative_pred_samples: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.dropoutcell_type = dropoutcell_type
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_cat = len(cardinality)
        self.scaling = scaling
        self.dtype = dtype
        self.nonnegative_pred_samples = nonnegative_pred_samples

        assert len(cardinality) == len(embedding_dimension), (
            "embedding_dimension should be a list with the same size as"
            " cardinality"
        )

        assert len(set(lags_seq)) == len(
            lags_seq
        ), "no duplicated lags allowed!"
        lags_seq.sort()

        self.lags_seq = lags_seq

        self.distr_output = distr_output
        RnnCell = {"lstm": mx.gluon.rnn.LSTMCell, "gru": mx.gluon.rnn.GRUCell}[
            self.cell_type
        ]

        self.target_shape = distr_output.event_shape

        # TODO: is the following restriction needed?
        assert (
            len(self.target_shape) <= 1
        ), "Argument `target_shape` should be a tuple with 1 element at most"

        Dropout = {
            "ZoneoutCell": ZoneoutCell,
            "RNNZoneoutCell": RNNZoneoutCell,
            "VariationalDropoutCell": VariationalDropoutCell,
            "VariationalZoneoutCell": VariationalZoneoutCell,
        }[self.dropoutcell_type]

        with self.name_scope():
            self.proj_distr_args = distr_output.get_args_proj()
            self.rnn = mx.gluon.rnn.HybridSequentialRNNCell()
            for k in range(num_layers):
                cell = RnnCell(hidden_size=num_cells)
                cell = mx.gluon.rnn.ResidualCell(cell) if k > 0 else cell
                # we found that adding dropout to outputs doesn't improve the
                # performance, so we only drop states
                if "Zoneout" in self.dropoutcell_type:
                    cell = (
                        Dropout(cell, zoneout_states=dropout_rate)
                        if dropout_rate > 0.0
                        else cell
                    )
                elif "Dropout" in self.dropoutcell_type:
                    cell = (
                        Dropout(cell, drop_states=dropout_rate)
                        if dropout_rate > 0.0
                        else cell
                    )
                self.rnn.add(cell)
            self.rnn.cast(dtype=dtype)
            self.embedder = FeatureEmbedder(
                cardinalities=cardinality,
                embedding_dims=embedding_dimension,
                dtype=self.dtype,
            )

            if scaling:
                self.scaler = MeanScaler(
                    keepdims=True,
                    default_scale=default_scale,
                    minimum_scale=minimum_scale,
                )
            else:
                self.scaler = NOPScaler(keepdims=True)

            self.num_imputation_samples = num_imputation_samples

            # Switch between vanilla mode and imputing the missing values
            # with the model during training/prediction
            if impute_missing_values:
                self.unroll_encoder = self.unroll_encoder_imputation
                self.include_zeros_in_denominator = True
            else:
                self.unroll_encoder = self.unroll_encoder_default
                self.include_zeros_in_denominator = False

    @staticmethod
    def get_lagged_subsequences(
        F,
        sequence: Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length : int
            length of sequence in the T (time) dimension (axis = 1).
        indices : List[int]
            list of lag indices to be used.
        subsequences_length : int
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: sequence_length - lag_index - subsequences_length >= 0
        # for all lag_index, hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            "lags cannot go further than history length, "
            f"found lag {max(indices)} while history length is only "
            f"{sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(
                F.slice_axis(
                    sequence, axis=1, begin=begin_index, end=end_index
                )
            )

        return F.stack(*lagged_values, axis=-1)

    def imputation_rnn_unroll(
        self,
        F,
        begin_state: List[Tensor],
        sequence: Tensor,
        sequence_length: int,
        subsequences_length: int,
        scale: Tensor,
        target: Tensor,
        target_observed_values: Tensor,
        time_feat: Tensor,
        repeated_static_feat: Tensor,
        is_padded_indicator: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Unrolls the RNN and imputes missing values with samples from the
        current model such that later unrolling steps get model imputed values
        as lags.

        Parameters
        ----------
        F
        begin_state
            Begin state of the RNN.
        sequence
            Target sequence. (history_length + prediction_length)
            during training, history_length else.
        sequence_length
            Length of the target sequence. (history_length + prediction_length)
            during training, history_length else.
        subsequences_length
            Length of the subsequence (context_length + prediction_length)
            during training, context_length else.
        scale
            Tensor containing the time series scales.
        target
            Target sequence of subsequence length.
        target_observed_values
            Observed value indicators of the target tensor.
        time_feat
            Time features.
        repeated_static_feat
            Repeated static cat features.
        is_padded_indicator
            Indicator whether the sequence is padded.

        Returns
        -------
        RNN output tensor, state, and sequence with imputed values
        """
        pass

    def prepare_inputs_imputation_step(
        self,
        F,
        begin_state: List[Tensor],
        imputed_sequence: Tensor,
        sequence_length: int,
        subsequences_length: int,
        scale: Tensor,
        target: Tensor,
        target_observed_values: Tensor,
        time_feat: Tensor,
        repeated_static_feat: Tensor,
        is_padded_indicator: Tensor,
        state,
        i: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Prepares inputs for the next LSTM unrolling step at step i.
        """
        pass

    def impute_target_if_unobserved(
        self,
        F,
        output,
        scale,
        current_target,
        current_observed_indicator,
        is_pad,
    ) -> Tensor:
        """
        This will impute the target at unrolling step i if the value is not
        observed. If the target value is a padded dummy value, it will be set
        to zero. We will keep the target value otherwise.

        Parameters
        ----------
        F
        output
            RNN outputs to construct the distribution at unrolling step i.
        scale
            Scale of the time series.
        current_target
            Tensor containing the current target.
        current_observed_indicator
            Tensor containing the current observed value indicator
        is_pad
            Tensor containing the current padding indicator.

        Returns
        -------
        Target (imputed/zero if unobserved).
        """
        pass


    def unroll_encoder_imputation(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        feat_static_real: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,  # (batch_size, history_length, num_features)
        past_target: Tensor,  # (batch_size, history_length, *target_shape)
        # (batch_size, history_length, *target_shape)
        past_observed_values: Tensor,
        past_is_pad: Tensor,  # (batch_size, history_length, *target_shape)
        future_observed_values: Optional[
            Tensor
        ],  # (batch_size, history_length, *target_shape)
        future_time_feat: Optional[
            Tensor
        ],  # (batch_size, prediction_length, num_features)
        future_target: Optional[
            Tensor
        ],  # (batch_size, prediction_length, *target_shape)
    ) -> Tuple[Tensor, List, Tensor, Tensor, Tensor]:
        """
        Unrolls the RNN encoder in "imputation mode" which will fill imputed
        values with samples from the DeepAR model.
        """
        pass

    def unroll_encoder_default(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        feat_static_real: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,  # (batch_size, history_length, num_features)
        past_target: Tensor,  # (batch_size, history_length, *target_shape)
        # (batch_size, history_length, *target_shape)
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_observed_values: Optional[Tensor],
        future_time_feat: Optional[
            Tensor
        ],  # (batch_size, prediction_length, num_features)
        future_target: Optional[
            Tensor
        ],  # (batch_size, prediction_length, *target_shape)
    ) -> Tuple[Tensor, List, Tensor, Tensor, Tensor]:
        """
        Unrolls the LSTM encoder over past and, if present, future data.

        Returns outputs and state of the encoder, plus the scale of past_target
        and a vector of static features that was constructed and fed as input
        to the encoder. All tensor arguments should have NTC layout.
        """
        pass

    def post_process_samples(self, F, samples: Tensor) -> Tensor:
        """
        Method to enforce domain-specific constraints on the generated samples.
        For example, we can enforce forecasts to be nonnegative.
        Parameters
        ----------
        samples
            Tensor of shape (batch_size * num_samples, 1)
        Returns
        -------
            Tensor of samples with the same shape.
        """

        if self.nonnegative_pred_samples:
            return F.relu(samples)

        return samples


class DeepARTrainingNetwork(DeepARNetwork):
    @validated()
    def __init__(self, alpha: float = 0, beta: float = 0, **kwargs) -> None:
        super().__init__(**kwargs)

        # regularization weights
        self.alpha = alpha
        self.beta = beta

        if alpha:
            self.ar_loss = ActivationRegularizationLoss(
                alpha, time_axis=1, batch_axis=0
            )
        if beta:
            self.tar_loss = TemporalActivationRegularizationLoss(
                beta, time_axis=1, batch_axis=0
            )

    def distribution(
        self,
        feat_static_cat: Tensor,
        feat_static_real: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Tensor,
        future_time_feat: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
        return_rnn_outputs: bool = False,
    ) -> Union[Distribution, Tuple[Distribution, Tensor]]:
        """
        Returns the distribution predicted by the model on the range of
        past_target and future_target.

        The distribution is obtained by unrolling the network with the true
        target, this is also the distribution that is being minimized during
        training. This can be used in anomaly detection, see for instance
        examples/anomaly_detection.py.

        Input arguments are the same as for the hybrid_forward method.

        Returns
        -------
        Distribution
            a distribution object whose mean has shape:
            (batch_size, context_length + prediction_length).
        Tensor
            (optional) when return_rnn_outputs=True, rnn_outputs will be
            returned so that it could be used for regularization
        """
        # unroll the decoder in "training mode"
        # i.e. by providing future data as well
        F = getF(feat_static_cat)
        rnn_outputs, _, scale, _, _ = self.unroll_encoder(
            F=F,
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_observed_values=future_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )

        distr_args = self.proj_distr_args(rnn_outputs)

        # return the output of rnn layers if return_rnn_outputs=True, so that
        # it can be used for regularization later assume no dropout for
        # outputs, so can be directly used for activation regularization
        return (
            (
                self.distr_output.distribution(distr_args, scale=scale),
                rnn_outputs,
            )
            if return_rnn_outputs
            else self.distr_output.distribution(distr_args, scale=scale)
        )

    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        feat_static_real: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_is_pad: Optional[Tensor],
        future_time_feat: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        """
        Computes the loss for training DeepAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        F
        feat_static_cat : (batch_size, num_features)
        feat_static_real : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape,
            seq_len)
        future_time_feat : (batch_size, prediction_length, num_features)
        future_target : (batch_size, prediction_length, *target_shape)
        future_observed_values : (batch_size, prediction_length, *target_shape)

        Returns loss with shape (batch_size, context + prediction_length, 1)
        -------
        """
        pass


class DeepARPredictionNetwork(DeepARNetwork):
    @validated()
    def __init__(self, num_parallel_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one, at the first time-step
        # of the decoder a lag of one corresponds to the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        F,
        static_feat: Tensor,
        past_target: Tensor,
        time_feat: Tensor,
        scale: Tensor,
        begin_states: List,
    ) -> Tensor:
        """
        Computes sample paths by unrolling the LSTM starting with a initial
        input and state.

        Parameters
        ----------
        static_feat : Tensor
            static features. Shape: (batch_size, num_static_features).
        past_target : Tensor
            target history. Shape: (batch_size, history_length).
        time_feat : Tensor
            time features. Shape: (batch_size, prediction_length,
            num_time_features).
        scale : Tensor
            tensor containing the scale of each element in the batch.
            Shape: (batch_size, 1, 1).
        begin_states : List
            list of initial states for the LSTM layers. The shape of each
            tensor of the list should be (batch_size, num_cells)
        Returns
        --------
        Tensor
            A tensor containing sampled paths.
            Shape: (batch_size, num_sample_paths, prediction_length).
        """

        # blows-up the dimension of each tensor to batch_size *
        # self.num_parallel_samples for increasing parallelism
        repeated_past_target = past_target.repeat(
            repeats=self.num_parallel_samples, axis=0
        )
        repeated_time_feat = time_feat.repeat(
            repeats=self.num_parallel_samples, axis=0
        )
        repeated_static_feat = static_feat.repeat(
            repeats=self.num_parallel_samples, axis=0
        ).expand_dims(axis=1)
        repeated_scale = scale.repeat(
            repeats=self.num_parallel_samples, axis=0
        )
        repeated_states = [
            s.repeat(repeats=self.num_parallel_samples, axis=0)
            for s in begin_states
        ]

        future_samples = []

        # for each future time-units we draw new samples for this time-unit and
        # update the state
        for k in range(self.prediction_length):
            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags = self.get_lagged_subsequences(
                F=F,
                sequence=repeated_past_target,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            # (batch_size * num_samples, 1, *target_shape, num_lags)
            lags_scaled = F.broadcast_div(
                lags, repeated_scale.expand_dims(axis=-1)
            )

            # from (batch_size * num_samples, 1, *target_shape, num_lags)
            # to (batch_size * num_samples, 1, prod(target_shape) * num_lags)
            input_lags = F.reshape(
                data=lags_scaled,
                shape=(-1, 1, prod(self.target_shape) * len(self.lags_seq)),
            )

            # (batch_size * num_samples, 1, prod(target_shape) * num_lags +
            # num_time_features + num_static_features)
            decoder_input = F.concat(
                input_lags,
                repeated_time_feat.slice_axis(axis=1, begin=k, end=k + 1),
                # observed_values.expand_dims(axis=1),
                repeated_static_feat,
                dim=-1,
            )

            # output shape: (batch_size * num_samples, 1, num_cells)
            # state shape: (batch_size * num_samples, num_cells)
            rnn_outputs, repeated_states = self.rnn.unroll(
                inputs=decoder_input,
                length=1,
                begin_state=repeated_states,
                layout="NTC",
                merge_outputs=True,
            )

            distr_args = self.proj_distr_args(rnn_outputs)

            # compute likelihood of target given the predicted parameters
            distr = self.distr_output.distribution(
                distr_args, scale=repeated_scale
            )

            # (batch_size * num_samples, 1, *target_shape)
            new_samples = distr.sample(dtype=self.dtype)
            new_samples = self.post_process_samples(F, new_samples)

            # (batch_size * num_samples, seq_len, *target_shape)
            repeated_past_target = F.concat(
                repeated_past_target, new_samples, dim=1
            )

            future_samples.append(new_samples)

        # (batch_size * num_samples, prediction_length, *target_shape)
        samples = F.concat(*future_samples, dim=1)

        # (batch_size, num_samples, prediction_length, *target_shape)
        return samples.reshape(
            shape=(
                (-1, self.num_parallel_samples)
                + (self.prediction_length,)
                + self.target_shape
            )
        )

    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        feat_static_real: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,  # (batch_size, history_length, num_features)
        past_target: Tensor,  # (batch_size, history_length, *target_shape)
        # (batch_size, history_length, *target_shape)
        past_observed_values: Tensor,
        # (batch_size, prediction_length, num_features)
        future_time_feat: Tensor,
        past_is_pad: Tensor,
    ) -> Tensor:
        """
        Predicts samples, all tensors should have NTC layout.
        Parameters
        ----------
        F
        feat_static_cat : (batch_size, num_features)
        feat_static_real : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape)
        future_time_feat : (batch_size, prediction_length, num_features)

        Returns
        -------
        Tensor
            Predicted samples
        """
        pass
