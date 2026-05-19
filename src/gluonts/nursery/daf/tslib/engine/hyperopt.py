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


from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Tuple, List, Union
from collections import OrderedDict
from pathlib import Path
from itertools import product
from functools import reduce
import traceback
import textwrap
import socket
import json

import torch as pt
import numpy as np
import pandas as pd

from .distributed import is_main_process, synchronize

if TYPE_CHECKING:
    from ..dataset import MetaDataset


class HyperOptManager(object):
    def __init__(
        self,
        dataset: MetaDataset,
        work_dir: Path,
        fixed_params: Dict,
        cuda_device: int,
        key_factor: str = "loss",
        min_mode: bool = True,
        resume: bool = False,
    ):
        self.dataset = dataset
        self.param_names, self.varied_params = zip(
            *sorted(dataset.hyperparam_space.items(), key=lambda x: x[0])
        )
        self.fixed_params = fixed_params
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cuda_device = cuda_device
        self.key_factor = key_factor
        self.min_mode = min_mode

        self.exp_ids = OrderedDict()
        self.records = OrderedDict()
        if resume:
            self.load_records()

    def __getitem__(self, key: Union[Tuple, int]) -> Union[int, Tuple]:
        if isinstance(key, tuple):
            index = 0
            for i, param in enumerate(key):
                idx = self.varied_params[i].index(param)
                index = index * len(self.varied_params[i]) + idx
            return index
        else:
            params = []
            for vp in reversed(self.varied_params):
                idx = key % len(vp)
                params.insert(0, vp[idx])
                key = key // len(vp)
            return tuple(params)

    def __len__(self):
        return reduce(lambda p, vp: p * len(vp), self.varied_params, 1)

    def hyperparameters(self, n_iter: int):
        n_choices_per_param = [len(vp) for vp in self.varied_params]
        n_choices = reduce(lambda p, x: p * x, n_choices_per_param, 1)
        n_covered = len(self.records)
        if n_iter > n_choices - n_covered:
            raise ValueError(
                f"# iterations {n_iter} exceeds the remaining grids {n_choices-n_covered}"
            )
        samples = list(np.random.choice(n_choices, n_iter + n_covered))
        for params in self.records:
            index = self[params]
            if index in samples:
                samples.remove(index)
        samples = samples[:n_iter]
        for index in samples:
            yield self[index]




    def load_records(self):
        print(f"Loading records from {self.work_dir}")
        record_path = self.work_dir.joinpath("records.csv")
        if record_path.exists():
            df = pd.read_csv(record_path, index_col=0)
            for exp_id, record in df.iterrows():
                params = (
                    record.loc[list(self.param_names)]
                    .values.flatten()
                    .tolist()
                )
                params = tuple(
                    eval(x) if isinstance(x, str) else x for x in params
                )
                self.exp_ids[params] = exp_id
                self.records[params] = record.iloc[
                    len(self.param_names) :
                ].to_dict()
            return
        raise RuntimeError("Cannot load previous checkpoint in random search.")



    def run_training(self, params: Dict) -> Dict:
        raise NotImplementedError

    def run_test(self, exp_dir: Path):
        raise NotImplementedError


