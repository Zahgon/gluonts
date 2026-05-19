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


from typing import NamedTuple, Callable, Optional, Iterator
import re

from torch import Tensor
from torch._six import container_abcs, int_classes, string_classes
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch as pt

from ..engine.distributed import is_distributed, get_world_size

_default_collate_err_msg_format = (
    "_default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)
np_str_obj_array_pattern = re.compile(r"[SaUO]")


def _default_collate(batch):
    r"""
    Puts each data field into a tensor with outer dimension batch size.
    """
    pass


def copy_to_gpu(data, cuda_device: int, non_blocking: bool):
    if not pt.cuda.is_available():
        raise SystemError("GPU is not available!")
    if isinstance(data, pt.Tensor):
        return data.cuda(cuda_device, non_blocking=non_blocking)
    elif isinstance(data, container_abcs.Mapping):
        return {
            key: copy_to_gpu(val, cuda_device, non_blocking)
            for key, val in data.items()
        }
    elif isinstance(data, container_abcs.Sequence):
        return [copy_to_gpu(t, cuda_device, non_blocking) for t in data]
    elif data is None:
        return None
    else:
        raise ValueError(
            f"expected tensor, sequence or dictionary of tensors, received {type(data).__name__}"
        )


class MetaDataset(NamedTuple):
    """
    Dataset Split Manager. Possess train/valid/test datasets and provide data
    loaders.

    Parameters:
    --------------
    train_set: torch.utils.data.Dataset
        training dataset
    valid_set: torch.utils.data.Dataset or None
        validation dataset if provided
    test_set: torch.utils.data.Dataset
        test dataset
    collate_fn: callable
        function that accepts a batch of inputs and
        returns a stacked version for each data field
    always_validation: bool
        if True, use test data at request of validation loader;
        otherwise return None
    """

    train_set: Dataset
    valid_set: Optional[Dataset]
    test_set: Dataset
    collate_fn: Callable[..., Optional[Tensor]] = _default_collate
    always_validation: bool = False

    def __getattr__(self, key: str):
        return getattr(self.train_set, key)




    def _data_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        cuda_device: int,
        is_training: bool,
        n_workers: int,
        n_batches: Optional[int],
    ) -> Iterator:
        if is_distributed() and is_training:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            batch_size //= get_world_size()
            shuffle = False
        else:
            sampler = None
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=n_workers,
            collate_fn=self.collate_fn,
            pin_memory=cuda_device >= 0,
            drop_last=n_batches is not None,
        )
        if n_batches is None:
            n_batches = len(loader)
        iterator = iter(loader)
        for batch in range(n_batches):
            try:
                data = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                data = next(iterator)
            if cuda_device >= 0:
                data = copy_to_gpu(data, cuda_device, non_blocking=True)
            yield data



