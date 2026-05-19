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

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .distribution import (
    Distribution,
    _sample_multiple,
    getF,
    nans_like,
    softplus,
)
from .distribution_output import DistributionOutput


class StudentT(Distribution):
    r"""
    Student's t-distribution.

    Parameters
    ----------
    mu
        Tensor containing the means, of shape `(*batch_shape, *event_shape)`.
    sigma
        Tensor containing the standard deviations, of shape
        `(*batch_shape, *event_shape)`.
    nu
        Nonnegative tensor containing the degrees of freedom of the
        distribution, of shape `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, mu: Tensor, sigma: Tensor, nu: Tensor, F=None) -> None:
        self.mu = mu
        self.sigma = sigma
        self.nu = nu

    @property
    def F(self):
        return getF(self.mu)




    @property
    def mean(self) -> Tensor:
        return self.F.where(self.nu > 1.0, self.mu, nans_like(self.mu))


    def log_prob(self, x: Tensor) -> Tensor:
        mu, sigma, nu = self.mu, self.sigma, self.nu
        F = self.F

        nup1_half = (nu + 1.0) / 2.0
        part1 = 1.0 / nu * F.square((x - mu) / sigma)
        Z = (
            F.gammaln(nup1_half)
            - F.gammaln(nu / 2.0)
            - 0.5 * F.log(math.pi * nu)
            - F.log(sigma)
        )

        ll = Z - nup1_half * F.log1p(part1)
        return ll

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:

        return _sample_multiple(
            s,
            mu=self.mu,
            sigma=self.sigma,
            nu=self.nu,
            num_samples=num_samples,
        )



class StudentTOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"mu": 1, "sigma": 1, "nu": 1}
    distr_cls: type = StudentT

    @classmethod
    def domain_map(cls, F, mu, sigma, nu):
        sigma = F.maximum(softplus(F, sigma), cls.eps())
        nu = 2.0 + F.maximum(softplus(F, nu), cls.eps())
        return mu.squeeze(axis=-1), sigma.squeeze(axis=-1), nu.squeeze(axis=-1)

