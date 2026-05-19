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

from typing import Optional, Union, List
from pytorch_lightning import Callback, Trainer
import torch
from gluonts.time_feature import get_seasonality

from meta.models.module import MetaLightningModule
from meta.common.torch import tensor_to_np
from meta.metrics.numpy import compute_metrics


class QuantileMetricLoggerCallback(Callback):
    """
    A callback that computes additional metrics on a numpy representation of
    the dataset every n epochs. The computed values are logged to the output
    file of the pytorch lightning logger.

    Args:
        quantiles: The quantiles that are predicted.
        split: Specifies the split the batch comes from (i.e. from training or validation split)
        every_n_epochs: Specifies how often the plots are generated.
            Setting this to a large value can save time since plotting can be time consuming
            (especially when small datasets are used).
    """

    def __init__(
        self,
        quantiles: List[str],
        split: Optional[Union["train", "val"]] = None,
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.split = split
        self.every_n_epochs = every_n_epochs

