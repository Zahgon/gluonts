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

import numpy as np

import mxnet as mx

from gluonts.core.component import (
    equals,
    equals_default_impl,
    skip_encoding,
    tensor_to_numpy,
)


@equals.register(mx.gluon.ParameterDict)
def equals_parameter_dict(
    this: mx.gluon.ParameterDict, that: mx.gluon.ParameterDict
) -> bool:
    """Structural equality check between two
    :class:`~mxnet.gluon.ParameterDict` objects.

    Two parameter dictionaries ``this`` and ``that`` are considered
    *structurally equal* if the following conditions are satisfied:

    1. They contain the same keys (modulo the key prefix which is stripped).
    2. The data in the corresponding value pairs is equal, as defined by the
       :func:`~mxnet.test_utils.almost_equal` function (in this case we call
       the function with ``equal_nan=True``, that is, two aligned ``NaN``
       values are always considered equal).

    Specializes :func:`equals` for invocations where the first parameter is an
    instance of the :class:`~mxnet.gluon.ParameterDict` class.

    Parameters
    ----------
    this, that
        Objects to compare.

    Returns
    -------
    bool
        A boolean value indicating whether ``this`` and ``that`` are
        structurally equal.

    See Also
    --------
    equals
        Dispatching function.
    """
    pass


@equals.register(mx.gluon.HybridBlock)
def equals_representable_block(
    this: mx.gluon.HybridBlock, that: mx.gluon.HybridBlock
) -> bool:
    """
    Structural equality check between two :class:`~mxnet.gluon.HybridBlock`
    objects with :func:`validated` initializers.

    Two blocks ``this`` and ``that`` are considered *structurally equal* if all
    the conditions of :func:`equals` are met, and in addition their parameter
    dictionaries obtained with
    :func:`~mxnet.gluon.block.Block.collect_params` are also structurally
    equal.

    Specializes :func:`equals` for invocations where the first parameter is an
    instance of the :class:`~mxnet.gluon.HybridBlock` class.

    Parameters
    ----------
    this, that
        Objects to compare.

    Returns
    -------
    bool
        A boolean value indicating whether ``this`` and ``that`` are
        structurally equal.

    See Also
    --------
    equals
        Dispatching function.
    equals_parameter_dict
        Specialization of :func:`equals` for Gluon
        :class:`~mxnet.gluon.ParameterDict` input arguments.
    """
    pass




@tensor_to_numpy.register(mx.ndarray.NDArray)
def _(tensor: mx.ndarray.NDArray) -> np.ndarray:
    return tensor.asnumpy()
