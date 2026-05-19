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

import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset, DataEntry, Dataset


class Grouper:
    # todo the contract of this grouper is missing from the documentation, what it does when, how it pads values etc
    def __init__(
        self,
        fill_value: float = 0.0,
        max_target_dim: int = None,
        align_data: bool = True,
        num_test_dates: int = None,
    ) -> None:
        self.fill_value = fill_value

        self.first_timestamp = pd.Timestamp(2200, 1, 1, 12)
        self.last_timestamp = pd.Timestamp(1800, 1, 1, 12)
        self.frequency = None
        self.align_data = align_data
        self.max_target_length = 0
        self.num_test_dates = num_test_dates
        self.max_target_dimension = max_target_dim

    def __call__(self, dataset: Dataset) -> Dataset:
        self._preprocess(dataset)
        return self._group_all(dataset)




    def _preprocess(self, dataset: Dataset) -> None:
        """
        The preprocess function iterates over the dataset to gather data that
        is necessary for grouping.

        This includes:
             1. Storing first/last timestamp in the dataset
             2. Aligning time series
             3. Calculating groups
        """
        for data in dataset:
            timestamp = data["start"]
            self.first_timestamp = min(self.first_timestamp, timestamp)

            self.frequency = (
                timestamp.freq if self.frequency is None else self.frequency
            )
            self.last_timestamp = max(
                self.last_timestamp,
                timestamp + len(data["target"]) * self.frequency,
            )

            # todo
            self.max_target_length = max(
                self.max_target_length, len(data["target"])
            )
        logging.info(
            f"first/last timestamp found: {self.first_timestamp}/{self.last_timestamp}"
        )


