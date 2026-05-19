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

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.mx import Tensor
from gluonts.mx.context import get_mxnet_context

from .binning_helpers import (
    bin_edges_from_bin_centers,
    ensure_binning_monotonicity,
)
from .representation import Representation


class GlobalRelativeBinning(Representation):
    """
    A class representing a global relative binning approach. This binning first
    rescales all input series by their respective mean (relative) and then
    performs one binning across all series (global).

    Parameters
    ----------
    num_bins
        The number of discrete bins/buckets that we want values to be mapped
        to.
        (default: 1024)
    is_quantile
        Whether the binning is quantile or linear. Quantile binning allocated
        bins based on the cumulative distribution function, while linear
        binning allocates evenly spaced bins.
        (default: True, i.e. quantile binning)

    linear_scaling_limit
        The linear scaling limit. Values which are larger than
        linear_scaling_limit times the mean will be capped at
        linear_scaling_limit.
        (default: 10)
    quantile_scaling_limit
        The quantile scaling limit. Values which are larger than the quantile
        evaluated at quantile_scaling_limit will be capped at the quantile
        evaluated at quantile_scaling_limit.
        (default: 0.99)
    """

    @validated()
    def __init__(
        self,
        num_bins: int = 1024,
        is_quantile: bool = True,
        linear_scaling_limit: int = 10,
        quantile_scaling_limit: float = 0.99,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_bins = num_bins
        self.is_quantile = is_quantile

        self.linear_scaling_limit = linear_scaling_limit
        self.quantile_scaling_limit = quantile_scaling_limit

        self.bin_edges = self.params.get_constant(
            "bin_edges", mx.nd.zeros(self.num_bins + 1)
        )
        self.bin_centers = self.params.get_constant(
            "bin_centers", mx.nd.zeros(self.num_bins)
        )




