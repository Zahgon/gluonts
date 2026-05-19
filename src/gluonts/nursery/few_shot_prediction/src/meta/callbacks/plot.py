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
import matplotlib.pyplot as plt

from meta.data.batch import TripletBatch
from meta.models.module import MetaLightningModule
from .common import get_save_dir_from_csvlogger, get_loss_steps
from meta.data.batch import SeriesBatch
from meta.vis.forecast import (
    plot_quantile_forecast,
    plot_forecast_supportset_attention,
)


class ForecastPlotLoggerCallback(Callback):
    """
    A callback that stores plots of the  predictions for a collection of
    samples every n epochs. The plots display the query (past and future) and
    forecasted quantiles. This callback is intended for models without support
    set and attention mechanism.

    Args:
        log_batch: For each sample in the batch the prediction is plotted when the callback is called.
            The batch size should not be too large (i.e. < 10).
        quantiles: The quantiles that are predicted.
        split: Specifies the split the batch comes from (i.e. from training or validation split)
        every_n_epochs: Specifies how often the plots are generated.
            Setting this to a large value can save time since plotting can be time consuming
            (especially when small datasets are used).
    """

    def __init__(
        self,
        log_batch: TripletBatch,
        quantiles: List[str],
        split: Optional[Union["train", "val"]] = None,
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.log_batch = log_batch
        self.quantiles = quantiles
        self.split = split
        self.every_n_epochs = every_n_epochs



class ForecastSupportSetAttentionPlotLoggerCallback(Callback):
    """
    A callback that stores plots of the  predictions for a collection of
    samples every n epochs. The plots display the query (past and future),
    forecasted quantiles and the time series in the support set of this sample
    aligned with their attention scores. This callback works only for models
    with attention mechanism!

    Args:
        log_batch: For each sample in the batch the prediction is plotted when the callback is called.
            The batch size should not be too large (i.e. < 10).
        quantiles: The quantiles that are predicted.
        split: Specifies the split the batch comes from (i.e. from training or validation split)
        every_n_epochs: Specifies how often the plots are generated.
            Setting this to a large value can save time since plotting can be time consuming
            (especially when small datasets are used).
    """

    def __init__(
        self,
        log_batch: TripletBatch,
        quantiles: List[str],
        split: Optional[Union["train", "val"]] = None,
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.log_batch = log_batch
        self.quantiles = quantiles
        self.split = split
        self.every_n_epochs = every_n_epochs



class LossPlotLoggerCallback(Callback):
    """
    A callback that stores plots of the training and macro-averaged validation
    loss curve every n epochs.

    Args:
        every_n_epochs: Specifies how often the plots are generated.
            Setting this to a large value can save time since plotting can be time consuming
            (especially when small datasets are used).
    """

    def __init__(self, every_n_epochs: int = 1):
        super().__init__()
        self.every_n_epochs = every_n_epochs





class CheatLossPlotLoggerCallback(LossPlotLoggerCallback):
    """
    A callback that stores plots of the training and multiple validation losses
    curve every n epochs.

    Args:
        every_n_epochs: Specifies how often the plots are generated.
            Setting this to a large value can save time since plotting can be time consuming
            (especially when small datasets are used).
        dataset_names_val: The names of the datasets that the validation loss should be plotted for.
    """

    def __init__(self, dataset_names_val, **kwargs):
        super().__init__(**kwargs)
        self.dataset_names_val = dataset_names_val



class MacroCRPSPlotCallback(Callback):
    """
    A callback that stores plots of the validation losses and a macro-averaged
    validation loss curve every n epochs.

    Args:
        every_n_epochs: Specifies how often the plots are generated.
            Setting this to a large value can save time since plotting can be time consuming
            (especially when small datasets are used).
    """

    def __init__(
        self,
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        # We only compute this on the data as it is (no rescaling)



