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

from typing import Any, Dict, Optional, List
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn
from torchmetrics import Metric
from torch.optim.lr_scheduler import ReduceLROnPlateau

from meta.data.batch import TripletBatch
from meta.models.model import MetaModel
from meta.common.torch import get_mask


class MetaLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for the meta models.
    """

    def __init__(
        self,
        model: MetaModel,
        val_dataset_names: List[str],
        test_dataset_names: List[str],
        lr_scheduler_monitor: str,
        loss: Metric,
        crps: Metric,
        crps_scaled: Optional[Metric] = None,
        lr: float = 1e-3,
        lr_on_plateau_patience: int = 5,
        lr_on_plateau_factor: float = 0.3,
        quantile_width: Metric = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.train_loss = loss.clone()
        self.lr = lr
        self.lr_on_plateau_patience = lr_on_plateau_patience
        self.lr_on_plateau_factor = lr_on_plateau_factor
        self.lr_scheduler_monitor = lr_scheduler_monitor
        self.quantile_width = quantile_width

        # validation
        self.val_loss = nn.ModuleList(
            [loss.clone() for _ in range(len(val_dataset_names))]
        )
        self.val_loss_macro = sum(self.val_loss) / len(self.val_loss)

        if quantile_width is not None:
            self.val_quantile_width = nn.ModuleList(
                [quantile_width.clone() for _ in range(len(val_dataset_names))]
            )
            self.test_quantile_width = nn.ModuleList(
                [
                    quantile_width.clone()
                    for _ in range(len(test_dataset_names))
                ]
            )

        self.val_dataset_names = val_dataset_names
        (
            self.val_crps,
            self.val_crps_macro,
            self.val_crps_scaled,
            self.val_crps_scaled_macro,
        ) = _init_metrics(val_dataset_names, crps, crps_scaled)

        # test
        self.test_dataset_names = test_dataset_names
        (
            self.test_crps,
            self.test_crps_macro,
            self.test_crps_scaled,
            self.test_crps_scaled_macro,
        ) = _init_metrics(test_dataset_names, crps, crps_scaled)

    def configure_optimizers(self) -> Dict:
        """
        Example of how to use StepLR:
            {
                "scheduler": StepLR(optimizer, gamma=0.99, ...)
                interval": "epoch",
                "frequency": 1,
            }
        """
        pass







def _init_metrics(
    dataset_names: List[str],
    crps: Metric,
    crps_scaled: Optional[Metric] = None,
):
    n_datasets = len(dataset_names)
    crps = nn.ModuleList([crps.clone() for _ in range(n_datasets)])
    macro_crps = sum(crps) / len(crps)
    if crps_scaled:
        crps_scaled = nn.ModuleList(
            [crps_scaled.clone() for _ in range(n_datasets)]
        )
        macro_crps_scaled = sum(crps_scaled) / len(crps_scaled)
    return (
        crps,
        macro_crps,
        crps_scaled,
        macro_crps_scaled if crps_scaled else None,
    )
