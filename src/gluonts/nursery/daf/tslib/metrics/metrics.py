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


import torch as pt
from torch import Tensor, BoolTensor
from torch.nn.functional import l1_loss, mse_loss
from torch.distributions import Distribution, Normal as Gaussian


def quantile_error(preds: Tensor, target: Tensor, percentage: float) -> Tensor:
    diff = target - preds
    weight = pt.where(
        condition=diff > 0,
        input=diff.new_tensor(percentage),
        other=diff.new_tensor(percentage - 1),
    )
    return diff * weight




def absolute_error(preds: Tensor, target: Tensor) -> Tensor:
    return quantile_error(preds, target, 0.5) * 2








