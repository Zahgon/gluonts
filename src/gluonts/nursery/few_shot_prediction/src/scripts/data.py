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

import os
from pathlib import Path
import itertools
import yaml
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import click
import torch

from meta.datasets import get_data_module, DATASETS_FILTERED
from meta.datasets.cheat import CheatMetaData
from meta.common.torch import tensor_to_np
import pandas as pd


@click.group()
def main():
    return


@main.command()
@click.option(
    "--catch22",
    default=False,
    help="Compute catch22 features and nearest neighbors. Can be time consuming!",
)
def real(catch22: bool):
    """
    Download and process real-world datasets specified in 'DATASETS_FILTERED'.

    Parameters
    ----------
    catch22: bool
        If true, catch22 features for each time series and 100 nearest neighbors w.r.t. l2 distance in
        feature space are computed.
    """
    pass


@main.command()
@click.argument("config", type=click.Path(exists=True), nargs=1)
def artificial(
    config: str,
):
    """
    Generate artificial datasets specified in a config file.

    Parameters
    ----------
    config: str
        Path to config file.
    """
    pass


@main.command()
def statistics():
    """
    Compute statistics of real-world datasets.
    """
    pass


if __name__ == "__main__":
    main()
