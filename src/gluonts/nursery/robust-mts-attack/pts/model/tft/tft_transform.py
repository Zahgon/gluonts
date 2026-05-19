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

from collections import Counter
from typing import Iterator, List, Optional

import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    InstanceSplitter,
    MapTransformation,
    shift_timestamp,
    target_transformation_length,
)
from gluonts.transform.sampler import InstanceSampler


class BroadcastTo(MapTransformation):
    @validated()
    def __init__(
        self,
        field: str,
        ext_length: int = 0,
        target_field: str = FieldName.TARGET,
    ) -> None:
        self.field = field
        self.ext_length = ext_length
        self.target_field = target_field



class TFTInstanceSplitter(InstanceSplitter):
    @validated()
    def __init__(
        self,
        instance_sampler: InstanceSampler,
        past_length: int,
        future_length: int,
        target_field: str = FieldName.TARGET,
        is_pad_field: str = FieldName.IS_PAD,
        start_field: str = FieldName.START,
        forecast_start_field: str = FieldName.FORECAST_START,
        observed_value_field: str = FieldName.OBSERVED_VALUES,
        lead_time: int = 0,
        output_NTC: bool = True,
        time_series_fields: List[str] = [],
        past_time_series_fields: List[str] = [],
        dummy_value: float = 0.0,
    ) -> None:
        super().__init__(
            target_field=target_field,
            is_pad_field=is_pad_field,
            start_field=start_field,
            forecast_start_field=forecast_start_field,
            instance_sampler=instance_sampler,
            past_length=past_length,
            future_length=future_length,
            lead_time=lead_time,
            output_NTC=output_NTC,
            time_series_fields=time_series_fields,
            dummy_value=dummy_value,
        )

        assert past_length > 0, "The value of `past_length` should be > 0"
        assert future_length > 0, "The value of `future_length` should be > 0"

        self.observed_value_field = observed_value_field
        self.past_ts_fields = past_time_series_fields

