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

from typing import Iterator, List, Optional, Tuple

import numpy as np
from pandas.tseries.offsets import BaseOffset

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.zebras._util import pad_axis

from ._base import FlatMapTransformation
from .sampler import ContinuousTimePointSampler, InstanceSampler


class InstanceSplitter(FlatMapTransformation):
    """
    Split instances from a dataset, by slicing the target and other time series
    fields at points in time selected by the specified sampler. The assumption
    is that all time series fields start at the same time point.

    It is assumed that time axis is always the last axis.

    The ``target_field`` and each field in ``time_series_fields`` are removed and
    replaced by two new fields, with prefix `past_` and `future_` respectively.

    A ``past_is_pad`` is also added, that indicates whether values at a given
    time point are padding or not.

    Parameters
    ----------

    target_field
        field containing the target
    is_pad_field
        output field indicating whether padding happened
    start_field
        field containing the start date of the time series
    forecast_start_field
        output field that will contain the time point where the forecast starts
    instance_sampler
        instance sampler that provides sampling indices given a time series
    past_length
        length of the target seen before making prediction
    future_length
        length of the target that must be predicted
    lead_time
        gap between the past and future windows (default: 0)
    output_NTC
        whether to have time series output in (time, dimension) or in
        (dimension, time) layout (default: True)
    time_series_fields
        fields that contains time series, they are split in the same interval
        as the target (default: None)
    dummy_value
        Value to use for padding. (default: 0.0)
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        instance_sampler: InstanceSampler,
        past_length: int,
        future_length: int,
        lead_time: int = 0,
        output_NTC: bool = True,
        time_series_fields: List[str] = [],
        dummy_value: float = 0.0,
    ) -> None:
        super().__init__()

        assert future_length > 0, "The value of `future_length` should be > 0"

        self.instance_sampler = instance_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.lead_time = lead_time
        self.output_NTC = output_NTC
        self.ts_fields = time_series_fields
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.dummy_value = dummy_value







class CanonicalInstanceSplitter(FlatMapTransformation):
    """
    Selects instances, by slicing the target and other time series like arrays
    at random points in training mode or at the last time point in prediction
    mode. Assumption is that all time like arrays start at the same time point.

    In training mode, the returned instances contain past_`target_field`
    as well as past_`time_series_fields`.

    In prediction mode, one can set `use_prediction_features` to get
    future_`time_series_fields`.

    If the target array is one-dimensional, the `target_field` in the resulting
    instance has shape (`instance_length`). In the multi-dimensional case, the
    instance has shape (`dim`, `instance_length`), where `dim` can also take a
    value of 1.

    In the case of insufficient number of time series values, the
    transformation also adds a field 'past_is_pad' that indicates whether
    values where padded or not, and the value is padded with
    `default_pad_value` with a default value 0.
    This is done only if `allow_target_padding` is `True`,
    and the length of `target` is smaller than `instance_length`.

    Parameters
    ----------
    target_field
        fields that contains time series
    is_pad_field
        output field indicating whether padding happened
    start_field
        field containing the start date of the time series
    forecast_start_field
        field containing the forecast start date
    instance_sampler
        instance sampler that provides sampling indices given a time series
    instance_length
        length of the target seen before making prediction
    output_NTC
        whether to have time series output in (time, dimension) or in
        (dimension, time) layout
    time_series_fields
        fields that contains time series, they are split in the same interval
        as the target
    allow_target_padding
        flag to allow padding
    pad_value
        value to be used for padding
    use_prediction_features
        flag to indicate if prediction range features should be returned
    prediction_length
        length of the prediction range, must be set if
        use_prediction_features is True
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        instance_sampler: InstanceSampler,
        instance_length: int,
        output_NTC: bool = True,
        time_series_fields: List[str] = [],
        allow_target_padding: bool = False,
        pad_value: float = 0.0,
        use_prediction_features: bool = False,
        prediction_length: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.instance_sampler = instance_sampler
        self.instance_length = instance_length
        self.output_NTC = output_NTC
        self.dynamic_feature_fields = time_series_fields
        self.target_field = target_field
        self.allow_target_padding = allow_target_padding
        self.pad_value = pad_value
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field

        assert (
            not use_prediction_features or prediction_length is not None
        ), "You must specify `prediction_length` if `use_prediction_features`"

        self.use_prediction_features = use_prediction_features
        self.prediction_length = prediction_length





class ContinuousTimeInstanceSplitter(FlatMapTransformation):
    """
    Selects training instances by slicing "intervals" from a continuous-time
    process instantiation. Concretely, the input data is expected to describe
    an instantiation from a point (or jump) process, with the "target"
    identifying inter-arrival times and other features (marks), as described in
    detail below.

    The splitter will then take random points in continuous time from each
    given observation, and return a (variable-length) array of points in
    the past (context) and the future (prediction) intervals.

    The transformation is analogous to its discrete counterpart
    `InstanceSplitter` except that

    - It does not allow "incomplete" records. That is, the past and future
      intervals sampled are always complete
    - Outputs a (T, C) layout.
    - Does not accept `time_series_fields` (i.e., only accepts target fields)
      as these would typically not be available in TPP data.

    The target arrays are expected to have (2, T) layout where the first axis
    corresponds to the (i) inter-arrival times between consecutive points, in
    order and (ii) integer identifiers of marks (from
    {0, 1, ..., :code:`num_marks`}). The returned arrays will have (T, 2)
    layout.

    For example, the array below corresponds to a target array where points
    with timestamps 0.5, 1.1, and 1.5 were observed belonging to categories
    (marks) 3, 1 and 0 respectively: :code:`[[0.5, 0.6, 0.4], [3, 1, 0]]`.

    Parameters
    ----------
    past_interval_length
        length of the interval seen before making prediction
    future_interval_length
        length of the interval that must be predicted
    train_sampler
        instance sampler that provides sampling indices given a time series
    target_field
        field containing the target
    start_field
        field containing the start date of the of the point process observation
    end_field
        field containing the end date of the point process observation
    forecast_start_field
        output field that will contain the time point where the forecast starts
    """

    @validated()
    def __init__(
        self,
        past_interval_length: float,
        future_interval_length: float,
        freq: BaseOffset,
        instance_sampler: ContinuousTimePointSampler,
        target_field: str = FieldName.TARGET,
        start_field: str = FieldName.START,
        end_field: str = "end",
        forecast_start_field: str = FieldName.FORECAST_START,
    ) -> None:
        super().__init__()

        assert (
            future_interval_length > 0
        ), "Prediction interval must have length greater than 0."

        self.instance_sampler = instance_sampler
        self.past_interval_length = past_interval_length
        self.future_interval_length = future_interval_length
        self.target_field = target_field
        self.start_field = start_field
        self.end_field = end_field
        self.forecast_start_field = forecast_start_field
        self.freq = freq




class TFTInstanceSplitter(InstanceSplitter):
    """
    Instance splitter used by the Temporal Fusion Transformer model.

    Unlike ``InstanceSplitter``, this class returns known dynamic features as a
    single tensor of shape [..., context_length + prediction_length, ...]
    without splitting it into past & future parts. Moreover, this class
    supports dynamic features that are known in the past.
    """

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
        observed_value_field: Optional[str] = FieldName.OBSERVED_VALUES,
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

        self.observed_value_field = observed_value_field
        self.past_ts_fields = past_time_series_fields

