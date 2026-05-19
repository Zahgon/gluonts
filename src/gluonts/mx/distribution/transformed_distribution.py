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

from typing import Any, List, Optional, Tuple

import mxnet as mx
import numpy as np
from mxnet import autograd

from gluonts.core.component import validated
from gluonts.mx import Tensor

from . import bijection as bij
from .distribution import Distribution, _index_tensor, getF
from .bijection import AffineTransformation


class TransformedDistribution(Distribution):
    r"""
    A distribution obtained by applying a sequence of transformations on top of
    a base distribution.
    """

    @validated()
    def __init__(
        self, base_distribution: Distribution, transforms: List[bij.Bijection]
    ) -> None:
        self.base_distribution = base_distribution
        self.transforms = transforms
        self.is_reparameterizable = self.base_distribution.is_reparameterizable

        # use these to cache shapes and avoid recomputing all steps
        # the reason we cannot do the computations here directly
        # is that this constructor would fail in mx.symbol mode
        self._event_dim: Optional[int] = None
        self._event_shape: Optional[Tuple] = None
        self._batch_shape: Optional[Tuple] = None

    @property
    def F(self):
        return self.base_distribution.F



    def __getitem__(self, item):
        bd_slice = self.base_distribution[item]
        trans_slice = [self._slice_bijection(t, item) for t in self.transforms]
        return TransformedDistribution(bd_slice, trans_slice)




    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        with autograd.pause():
            s = self.base_distribution.sample(
                num_samples=num_samples, dtype=dtype
            )
            for t in self.transforms:
                s = t.f(s)
            return s

    def sample_rep(
        self, num_samples: Optional[int] = None, dtype=float
    ) -> Tensor:
        s = self.base_distribution.sample_rep(
            num_samples=num_samples, dtype=dtype
        )
        for t in self.transforms:
            s = t.f(s)
        return s

    def log_prob(self, y: Tensor) -> Tensor:
        F = getF(y)
        lp = 0.0
        x = y
        for t in self.transforms[::-1]:
            x = t.f_inv(y)
            ladj = t.log_abs_det_jac(x, y)
            lp = lp - sum_trailing_axes(F, ladj, self.event_dim - t.event_dim)
            y = x

        return self.base_distribution.log_prob(x) + lp

    def cdf(self, y: Tensor) -> Tensor:
        x = y
        sign = 1.0
        for t in self.transforms[::-1]:
            x = t.f_inv(x)
            sign = sign * t.sign
        f = self.base_distribution.cdf(x)
        return sign * (f - 0.5) + 0.5

    def quantile(self, level: Tensor) -> Tensor:
        F = getF(level)

        sign = 1.0
        for t in self.transforms:
            sign = sign * t.sign

        if not isinstance(sign, (mx.nd.NDArray, mx.sym.Symbol)):
            level = level if sign > 0 else (1.0 - level)
            q = self.base_distribution.quantile(level)
        else:
            # level.shape = (#levels,)
            # q_pos.shape = (#levels, batch_size, ...)
            # sign.shape = (batch_size, ...)
            q_pos = self.base_distribution.quantile(level)
            q_neg = self.base_distribution.quantile(1.0 - level)
            cond = F.broadcast_greater(sign, sign.zeros_like())
            cond = F.broadcast_add(cond, q_pos.zeros_like())
            q = F.where(cond, q_pos, q_neg)

        for t in self.transforms:
            q = t.f(q)
        return q


class AffineTransformedDistribution(TransformedDistribution):
    @validated()
    def __init__(
        self,
        base_distribution: Distribution,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> None:
        super().__init__(base_distribution, [AffineTransformation(loc, scale)])

        self.loc = loc if loc is not None else 0
        self.scale = scale if scale is not None else 1

    @property
    def mean(self) -> Tensor:
        return self.base_distribution.mean * self.scale + self.loc



    # TODO: crps


def sum_trailing_axes(F, x: Tensor, k: int) -> Tensor:
    for _ in range(k):
        x = F.sum(x, axis=-1)
    return x
