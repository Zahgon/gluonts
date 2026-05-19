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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch.nn
import torch.optim

from .distr_tcn import DistributionalTCN


def train_step_from_batch(
    ts_chunks: torch.Tensor,
    targets: torch.Tensor,
    distr_tcn: DistributionalTCN,
    optimizer: torch.optim.Adam,
):
    """
    Arguments
    ----------
    ts_chunks: Mini-batch chunked from the time series
    targets: Corresponding chunk of target values
    distr_tcn: DistributionalTCN
    otimizer: Optimizer containing parameters, learning rate, etc
    """
    pass


def eval_on_series(
    distr_tcn: DistributionalTCN,
    optimizer: torch.optim.Adam,
    series_tensor: torch.Tensor,
    ts_len: int,
    context_length: int,
    is_train: bool = False,
    return_predictions: bool = False,
    lead_time: int = 1,
):
    """
    Arguments
    ----------
    distr_tcn: DistributionalTCN
    otimizer: Optimizer containing parameters, learning rate, etc
    series_tensor: Time series
    ts_len: Length of time series
    context_length: Number of time steps to input
    is_train: True if time series is training set
    return_predictions: True if to return (loss, predictions), False if to return loss only
    lead_time: Number of time steps to predict ahead
    """
    pass


def plot_prediction(
    val_ts_tensor: torch.Tensor,
    predictions: torch.Tensor,
    context_length: int,
    lead_time: int = 1,
    start: int = 0,
    end: int = 500,
    fig: Optional[matplotlib.figure.Figure] = None,
):
    """
    Arguments
    ----------
    val_ts_tensor: Time series
    predictions: Prediction series
    context_length: Number of time steps to input
    lead_time: Number of time steps to predict ahead
    start: Index of time series at which to start plotting
    end: Index of time series at which to end plotting
    """
    pass


def highlight_min(data, color="lightgreen"):
    """
    Highlights the minimum in a Series or DataFrame.
    """
    pass
