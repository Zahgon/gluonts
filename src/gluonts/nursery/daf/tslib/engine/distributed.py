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


from typing import Optional
import os

import torch as pt
from torch import Tensor
from torch import distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_distributed():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    if get_world_size() == 1:
        return
    dist.barrier()


def reduce_value(value: Tensor, dst: Optional[int] = 0, average: bool = True):
    world_size = get_world_size()
    if world_size < 2:
        return value
    if dst is None:
        dist.all_reduce(value)
        if average:
            value = value / world_size
    else:
        dist.reduce(value, dst)
        if (get_rank() == dst) and average:
            value = value / world_size
    return value


