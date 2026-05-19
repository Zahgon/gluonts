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

import itertools
import logging
import time
from typing import List, Optional
import mxnet as mx
import numpy as np
from gluonts.core.component import validated
from gluonts.dataset.loader import DataLoader
from gluonts.gluonts_tqdm import tqdm
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import HybridContext
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.metric import ndarray
from .callbacks import Callback, CallbackList

logger = logging.getLogger("gluonts_meta").getChild("trainer")

# make the IDE happy: mx.py does not explicitly import autograd
mx.autograd = autograd  # type: ignore


class TimedTrainer(Trainer):
    """
    A custom trainer whose training duration is based on wall clock time
    instead of epochs.
    """

    @validated()
    def __init__(
        self,
        training_time: float,
        validation_milestones: Optional[List[float]] = None,
        learning_rate: float = 1e-3,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        super().__init__(learning_rate=learning_rate)

        validation_milestones = validation_milestones or []
        assert all(
            x < y
            for x, y in zip(validation_milestones, validation_milestones[1:])
        ), "Validation milestones must be increasing."

        self.training_time = training_time
        self.validation_milestones = validation_milestones or []
        self.callbacks = CallbackList(callbacks or [])

    def __call__(
        self,
        net: nn.HybridBlock,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:
        logger.info("Start model training")
        net.initialize(ctx=self.ctx, init=self.init)

        with HybridContext(
            net=net,
            hybridize=self.hybridize,
            static_alloc=True,
            static_shape=True,
        ):
            self._train_loop(net, train_iter, validation_iter)



