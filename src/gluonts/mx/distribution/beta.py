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

from typing import Dict, List, Optional, Tuple

import numpy as np

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .distribution import Distribution, _sample_multiple, getF, softplus
from .distribution_output import DistributionOutput


class Beta(Distribution):
    r"""
    Beta distribution.

    Parameters
    ----------
    alpha
        Tensor containing the alpha shape parameters, of shape
        `(*batch_shape, *event_shape)`.
    beta
        Tensor containing the beta shape parameters, of shape
        `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, alpha: Tensor, beta: Tensor) -> None:
        self.alpha = alpha
        self.beta = beta

    @property
    def F(self):
        return getF(self.alpha)




    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F
        alpha, beta = self.alpha, self.beta

        return (
            (alpha - 1) * F.log(x)
            + (beta - 1) * F.log(1 - x)
            - F.gammaln(alpha)
            - F.gammaln(beta)
            + F.gammaln(alpha + beta)
        )

    @property
    def mean(self) -> Tensor:
        return self.alpha / (self.alpha + self.beta)



    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        epsilon = np.finfo(dtype).eps  # machine epsilon


        samples = _sample_multiple(
            s, alpha=self.alpha, beta=self.beta, num_samples=num_samples
        )

        return self.F.clip(data=samples, a_min=epsilon, a_max=1 - epsilon)



class BetaOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"alpha": 1, "beta": 1}
    distr_cls: type = Beta

    @classmethod
    def domain_map(cls, F, alpha, beta):
        r"""
        Maps raw tensors to valid arguments for constructing a Beta
        distribution.

        Parameters
        ----------
        F:
        alpha:
            Tensor of shape `(*batch_shape, 1)`
        beta:
            Tensor of shape `(*batch_shape, 1)`

        Returns
        -------
        Tuple[Tensor, Tensor]:
            Two squeezed tensors, of shape `(*batch_shape)`: both have entries
            mapped to the positive orthant.
        """
        alpha = F.maximum(softplus(F, alpha), cls.eps())
        beta = F.maximum(softplus(F, beta), cls.eps())
        return alpha.squeeze(axis=-1), beta.squeeze(axis=-1)


