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


from typing import TYPE_CHECKING, Optional

import torch as pt
from torch import Tensor, BoolTensor
from torch.nn import functional as F


class ZScoreNormalizer(object):
    def __init__(
        self,
        eps: float = 1e-5,
        rescale_loss: bool = True,
    ) -> None:
        self._buffers = {
            "offset": None,
            "scale": None,
        }
        self.eps = eps
        self.rescale_loss = rescale_loss




class LossFunction(object):
    def __call__(self, preds: Tensor, truth: Tensor) -> Tensor:
        raise NotImplementedError




class MSELoss(LossFunction):
    def __call__(self, preds: Tensor, truth: Tensor) -> Tensor:
        return F.mse_loss(preds, truth, reduction="none")



class MAELoss(LossFunction):
    def __call__(self, preds: Tensor, truth: Tensor) -> Tensor:
        return pt.abs(preds - truth)
