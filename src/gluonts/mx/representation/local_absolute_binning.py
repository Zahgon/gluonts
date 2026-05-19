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
from gluonts.mx import Tensor

from .binning_helpers import (
    bin_edges_from_bin_centers,
    ensure_binning_monotonicity,
)
from .representation import Representation


class LocalAbsoluteBinning(Representation):
    """
    A class representing a local absolute binning approach. This binning
    estimates a binning for every single time series on a local level and
    therefore implicitly acts as a scaling mechanism.

    Parameters
    ----------
    num_bins
        The number of discrete bins/buckets that we want values to be mapped
        to. (default: 1024)
    is_quantile
        Whether the binning is quantile or linear. Quantile binning allocated
        bins based on the cumulative distribution function, while linear
        binning allocates evenly spaced bins.(default: True, i.e. quantile
        binning)
    """

    @validated()
    def __init__(
        self, num_bins: int = 1024, is_quantile: bool = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.num_bins = num_bins
        self.is_quantile = is_quantile


