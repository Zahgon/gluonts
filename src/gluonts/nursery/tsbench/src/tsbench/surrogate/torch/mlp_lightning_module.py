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

# pylint: disable=arguments-differ,too-many-ancestors,abstract-method
from typing import List, Tuple
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch import nn, optim
from .losses import ListMLELoss


class MLPLightningModule(pl.LightningModule):
    """
    Lightning module which trains an MLP until convergence.
    """

    def __init__(
        self, model: nn.Module, loss: nn.Module, weight_decay: float = 0.0
    ):
        super().__init__()

        self.model = model
        self.loss = loss
        self.weight_decay = weight_decay
        self.uses_ranking = isinstance(self.loss, ListMLELoss)




