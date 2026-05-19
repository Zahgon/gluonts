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

from typing import Any

import numpy as np


from ._base import Kind, encode


@encode.register(np.dtype)
def encode_np_dtype(v: np.dtype) -> Any:
    """
    Specializes :func:`encode` for invocations where ``v`` is an instance of
    the :class:`~numpy.dtype` class.
    """
    pass


@encode.register(np.ndarray)
def encode_np_ndarray(v: np.ndarray) -> Any:
    """
    Specializes :func:`encode` for invocations where ``v`` is an instance of
    the :class:`~numpy.ndarray` class.
    """
    pass






