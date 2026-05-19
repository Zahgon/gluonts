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

import lightning.pytorch as pl
import torch

from gluonts.core.component import validated

from .module import WaveNet


class WaveNetLightningModule(pl.LightningModule):
    """
    LightningModule wrapper over WaveNet.

    Parameters
    ----------
    model_kwargs
        Keyword arguments to pass to WaveNet.
    lr, optional
        Learning rate, by default 1e-3
    weight_decay, optional
        Weight decay, by default 1e-8
    """

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = WaveNet(**model_kwargs)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        pass

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        pass

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        pass
