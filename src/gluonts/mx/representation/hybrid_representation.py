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

from .representation import Representation


class HybridRepresentation(Representation):
    """
    A class representing a hybrid approach of combining multiple
    representations into a single representation. Representations will be
    combined by concatenating them on dim=-1.

    Parameters
    ----------
    representations
        A list of representations. Elements must be of type Representation.
    """

    @validated()
    def __init__(self, representations: List, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representations = representations
        for representation in self.representations:
            self.register_child(representation)



