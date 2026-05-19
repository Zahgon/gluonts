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

from typing import List, Tuple

from mxnet.gluon import nn

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.block.cnn import CausalConv1D
from gluonts.mx.block.mlp import MLP
from gluonts.mx.block.rnn import RNN


class Seq2SeqEncoder(nn.HybridBlock):
    """
    Abstract class for the encoder.

    An encoder takes a `target` sequence with corresponding covariates and maps
    it into a static latent and a dynamic latent code with the same length as
    the `target` sequence.
    """

    def hybrid_forward(
        self,
        F,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        F:
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        target
            target time series,
            shape (batch_size, sequence_length)
        static_features
            static features,
            shape (batch_size, num_feat_static)
        dynamic_features
            dynamic_features,
            shape (batch_size, sequence_length, num_feat_dynamic)

        Returns
        -------
        Tensor
            static code,
            shape (batch_size, num_feat_static)
        Tensor
            dynamic code,
            shape (batch_size, sequence_length, num_feat_dynamic)
        """
        raise NotImplementedError

    def _assemble_inputs(
        self,
        F,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tensor:
        """
        Assemble features from target, static features, and the dynamic
        features.

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        target
            target time series,
            shape (batch_size, sequence_length, 1)
        static_features
            static features,
            shape (batch_size, num_feat_static)
        dynamic_features
            dynamic_features,
            shape (batch_size, sequence_length, num_feat_dynamic)

        Returns
        -------
        Tensor
            combined features,
            shape (batch_size, sequence_length,
                   num_feat_static + num_feat_dynamic + 1)
        """
        pass


# TODO: fix handling of static features
class HierarchicalCausalConv1DEncoder(Seq2SeqEncoder):
    """
    Defines a stack of dilated convolutions as the encoder.

    See the following paper for details:
    1. Van Den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O.,
    Graves, A., Kalchbrenner, N., Senior, A.W. and Kavukcuoglu, K., 2016,
    September. WaveNet: A generative model for raw audio. In SSW (p. 125).

    Parameters
    ----------
    dilation_seq
        dilation for each convolution in the stack.
    kernel_size_seq
        kernel size for each convolution in the stack.
    channels_seq
        number of channels for each convolution in the stack.
    use_residual
        flag to toggle using residual connections.
    use_static_feat
        flag to toggle whether to use use_static_feat as input to the encoder
    use_dynamic_feat
        flag to toggle whether to use use_dynamic_feat as input to the encoder
    """

    @validated()
    def __init__(
        self,
        dilation_seq: List[int],
        kernel_size_seq: List[int],
        channels_seq: List[int],
        use_residual: bool = False,
        use_static_feat: bool = False,
        use_dynamic_feat: bool = False,
        **kwargs,
    ) -> None:
        assert all(
            [x > 0 for x in dilation_seq]
        ), "`dilation_seq` values must be greater than zero"
        assert all(
            [x > 0 for x in kernel_size_seq]
        ), "`kernel_size_seq` values must be greater than zero"
        assert all(
            [x > 0 for x in channels_seq]
        ), "`channel_dim_seq` values must be greater than zero"

        super().__init__(**kwargs)

        self.use_residual = use_residual
        self.use_static_feat = use_static_feat
        self.use_dynamic_feat = use_dynamic_feat
        self.cnn = nn.HybridSequential()

        it = zip(channels_seq, kernel_size_seq, dilation_seq)
        for layer_no, (channels, kernel_size, dilation) in enumerate(it):
            convolution = CausalConv1D(
                channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                activation="relu",
                prefix=f"conv_{layer_no:#02d}'_",
            )
            self.cnn.add(convolution)

    def hybrid_forward(
        self,
        F,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        target
            target time series,
            shape (batch_size, sequence_length, 1)
        static_features
            static features,
            shape (batch_size, num_feat_static)
        dynamic_features
            dynamic_features,
            shape (batch_size, sequence_length, num_feat_dynamic)
        Returns
        -------
        Tensor
            static code,
            shape (batch_size, channel_seqs + (1) if use_residual)
        Tensor
            dynamic code,
            shape (batch_size, sequence_length, channel_seqs + (1) if
            use_residual)
        """
        pass


class RNNEncoder(Seq2SeqEncoder):
    """
    Defines RNN encoder that uses covariates and target as input to the RNN if
    desired.

    Parameters
    ----------
    mode
        type of the RNN. Can be either: rnn_relu (RNN with relu activation),
        rnn_tanh, (RNN with tanh activation), lstm or gru.
    hidden_size
        number of units per hidden layer.
    num_layers
        number of hidden layers.
    bidirectional
        toggle use of bi-directional RNN as encoder.
    use_static_feat
        flag to toggle whether to use use_static_feat as input to the encoder
    use_dynamic_feat
        flag to toggle whether to use use_dynamic_feat as input to the encoder
    """

    @validated()
    def __init__(
        self,
        mode: str,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        use_static_feat: bool = False,
        use_dynamic_feat: bool = False,
        **kwargs,
    ) -> None:
        assert num_layers > 0, "`num_layers` value must be greater than zero"
        assert hidden_size > 0, "`hidden_size` value must be greater than zero"

        super().__init__(**kwargs)

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_static_feat = use_static_feat
        self.use_dynamic_feat = use_dynamic_feat

        with self.name_scope():
            self.rnn = RNN(mode, hidden_size, num_layers, bidirectional)

    def hybrid_forward(
        self,
        F,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        target
            target time series,
            shape (batch_size, sequence_length, 1)
        static_features
            static features,
            shape (batch_size, num_feat_static)
        dynamic_features
            dynamic_features,
            shape (batch_size, sequence_length, num_feat_dynamic)

        Returns
        -------
        Tensor
            static code,
            shape (batch_size, num_feat_static)
        Tensor
            dynamic code,
            shape (batch_size, sequence_length, num_feat_dynamic)
        """
        pass


class MLPEncoder(Seq2SeqEncoder):
    """
    Defines a multilayer perceptron used as an encoder.

    Parameters
    ----------
    layer_sizes
        number of hidden units per layer.
    kwargs
    """

    @validated()
    def __init__(self, layer_sizes: List[int], **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = MLP(layer_sizes, flatten=True)

    def hybrid_forward(
        self,
        F,
        target: Tensor,
        static_features: Tensor,
        dynamic_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        target
            target time series,
            shape (batch_size, sequence_length)
        static_features
            static features,
            shape (batch_size, num_feat_static)
        dynamic_features
            dynamic_features,
            shape (batch_size, sequence_length, num_feat_dynamic)

        Returns
        -------
        Tensor
            static code,
            shape (batch_size, num_feat_static)
        Tensor
            dynamic code,
            shape (batch_size, sequence_length, num_feat_dynamic)
        """
        pass


class RNNCovariateEncoder(RNNEncoder):
    """
    Deprecated class only for compatibility; use RNNEncoder instead.
    """

    @validated()
    def __init__(
        self,
        use_static_feat: bool = True,
        use_dynamic_feat: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            use_static_feat=use_static_feat,
            use_dynamic_feat=use_dynamic_feat,
            **kwargs,
        )
