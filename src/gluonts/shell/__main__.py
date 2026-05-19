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

import logging
import traceback
from pathlib import Path
from typing import Optional

import click
import waitress

from gluonts.env import env as gluonts_env
from gluonts.shell.serve import Settings

from .env import ServeEnv, TrainEnv
from .exceptions import ForecasterNotFound
from .sagemaker import TrainPaths
from .util import Forecaster, forecaster_type_by_name

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    pass






if __name__ == "__main__":
    import logging
    import os

    from gluonts.env import env

    if "TRAINING_JOB_NAME" in os.environ:
        env._push(use_tqdm=False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
    )
    cli(prog_name=__package__)
