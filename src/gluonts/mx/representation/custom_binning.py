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

from .binning_helpers import bin_edges_from_bin_centers
from .representation import Representation


class CustomBinning(Representation):
    """
    A class representing binned representations with custom centers.

    Parameters
    ----------
    bin_centers
        The bins to be used to discretize the data.
        (default: 1024)
    """

    @validated()
    def __init__(self, bin_centers: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bin_edges = self.params.get_constant(
            "bin_edges", mx.nd.array(bin_edges_from_bin_centers(bin_centers))
        )
        self.bin_centers = self.params.get_constant(
            "bin_centers", mx.nd.array(bin_centers)
        )

        self.num_bins = len(bin_centers)




