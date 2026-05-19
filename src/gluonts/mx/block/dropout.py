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

from mxnet.gluon.rnn import (
    BidirectionalCell,
    ModifierCell,
    RecurrentCell,
    SequentialRNNCell,
)

from gluonts.core.component import validated
from gluonts.mx import Tensor


class VariationalZoneoutCell(ModifierCell):
    """
    Applies Variational Zoneout on base cell. The implementation follows.

    [GG16]_. Variational zoneout uses the same mask across time-steps. It can
    be applied to RNN outputs, and states. The masks for them are not shared.

    The mask is initialized when stepping forward for the first time and
    willremain the same until .reset() is called. Thus, if using the cell and
    stepping manually without calling .unroll(), the .reset() should be called
    after each sequence.

    Parameters
    ----------
    base_cell
        The cell on which to perform variational dropout.
    zoneout_outputs
        The dropout rate for outputs. Won't apply dropout if it equals 0.
    zoneout_states
        The dropout rate for state inputs on the first state channel.
        Won't apply dropout if it equals 0.
    """

    @validated()
    def __init__(
        self,
        base_cell: RecurrentCell,
        zoneout_outputs: float = 0.0,
        zoneout_states: float = 0.0,
    ):
        assert not isinstance(base_cell, BidirectionalCell), (
            "BidirectionalCell doesn't support zoneout since it doesn't"
            " support step. Please add VariationalZoneoutCell to the cells"
            " underneath instead."
        )
        assert (
            not isinstance(base_cell, SequentialRNNCell)
            or not base_cell._bidirectional
        ), (
            "Bidirectional SequentialRNNCell doesn't support zoneout. Please"
            " add VariationalZoneoutCell to the cells underneath instead."
        )
        super().__init__(base_cell)
        self.zoneout_outputs = zoneout_outputs
        self.zoneout_states = zoneout_states
        self._prev_output = None

        # shared masks across time-steps
        self.zoneout_states_mask = None
        self.zoneout_outputs_mask = None

    def __repr__(self):
        s = (
            "{name}(p_out={zoneout_outputs}, p_state={zoneout_states},"
            " {base_cell})"
        )
        return s.format(name=self.__class__.__name__, **self.__dict__)


    def reset(self):
        super().reset()
        self._prev_output = None

        self.zoneout_states_mask = None
        self.zoneout_outputs_mask = None





class RNNZoneoutCell(ModifierCell):
    """
    Applies Zoneout on base cell. The implementation follows [KMK16]_.

    Compared to mx.gluon.rnn.ZoneoutCell, this implementation uses the same
    mask for output and states[0], since for RNN cells, states[0] is the same
    as output, except for ResidualCell, where states[0] = input + ouptut

    Parameters
    ----------
    base_cell
        The cell on which to perform variational dropout.
    zoneout_outputs
        The dropout rate for outputs. Won't apply dropout if it equals 0.
    zoneout_states
        The dropout rate for state inputs on the first state channel.
        Won't apply dropout if it equals 0.
    """

    @validated()
    def __init__(
        self,
        base_cell: RecurrentCell,
        zoneout_outputs: float = 0.0,
        zoneout_states: float = 0.0,
    ):
        assert not isinstance(base_cell, BidirectionalCell), (
            "BidirectionalCell doesn't support zoneout since it doesn't"
            " support step. Please add RNNZoneoutCell to the cells underneath"
            " instead."
        )
        assert (
            not isinstance(base_cell, SequentialRNNCell)
            or not base_cell._bidirectional
        ), (
            "Bidirectional SequentialRNNCell doesn't support zoneout. "
            "Please add RNNZoneoutCell to the cells underneath instead."
        )
        super().__init__(base_cell)
        self.zoneout_outputs = zoneout_outputs
        self.zoneout_states = zoneout_states
        self._prev_output = None

    def __repr__(self):
        s = (
            "{name}(p_out={zoneout_outputs}, p_state={zoneout_states},"
            " {base_cell})"
        )
        return s.format(name=self.__class__.__name__, **self.__dict__)


    def reset(self):
        super().reset()
        self._prev_output = None

