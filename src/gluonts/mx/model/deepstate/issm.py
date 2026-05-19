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

from typing import List, Sequence, Tuple

from pandas.tseries.frequencies import to_offset

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.distribution.distribution import getF
from gluonts.mx.util import _broadcast_param
from gluonts.time_feature import (
    Constant as ZeroFeature,
    day_of_week_index,
    hour_of_day_index,
    minute_of_hour_index,
    month_of_year_index,
    TimeFeature,
    week_of_year_index,
    norm_freq_str,
)




def _make_2_block_diagonal(F, left: Tensor, right: Tensor) -> Tensor:
    """
    Creates a block diagonal matrix of shape (batch_size, m+n, m+n) where m and
    n are the sizes of the axis 1 of left and right respectively.

    Parameters
    ----------
    F
    left
        Tensor of shape (batch_size, seq_length, m, m)
    right
        Tensor of shape (batch_size, seq_length, n, n)
    Returns
    -------
    Tensor
        Block diagonal matrix of shape (batch_size, seq_length, m+n, m+n)
    """
    pass


class ISSM:
    r"""
    An abstract class for providing the basic structure of Innovation State
    Space Model (ISSM).

    The structure of ISSM is given by

        * dimension of the latent state
        * transition and innovation coefficients of the transition model
        * emission coefficient of the observation model
    """

    @validated()
    def __init__(self):
        pass

    def latent_dim(self) -> int:
        raise NotImplementedError

    def output_dim(self) -> int:
        raise NotImplementedError

    def time_features(self) -> List[TimeFeature]:
        raise NotImplementedError

    def emission_coeff(self, features: Tensor) -> Tensor:
        raise NotImplementedError

    def transition_coeff(self, features: Tensor) -> Tensor:
        raise NotImplementedError

    def innovation_coeff(self, features: Tensor) -> Tensor:
        raise NotImplementedError



class LevelISSM(ISSM):
    def latent_dim(self) -> int:
        return 1

    def output_dim(self) -> int:
        return 1

    def time_features(self) -> List[TimeFeature]:
        return [ZeroFeature()]





class LevelTrendISSM(LevelISSM):
    def latent_dim(self) -> int:
        return 2

    def output_dim(self) -> int:
        return 1

    def time_features(self) -> List[TimeFeature]:
        return [ZeroFeature()]



class SeasonalityISSM(LevelISSM):
    """
    Implements periodic seasonality which is entirely determined by the period
    `num_seasons`.
    """

    @validated()
    def __init__(self, num_seasons: int, time_feature: TimeFeature) -> None:
        super().__init__()
        self.num_seasons = num_seasons
        self.time_feature = time_feature

    def latent_dim(self) -> int:
        return self.num_seasons

    def output_dim(self) -> int:
        return 1

    def time_features(self) -> List[TimeFeature]:
        return [self.time_feature]




def MonthOfYearSeasonalISSM():
    return SeasonalityISSM(num_seasons=12, time_feature=month_of_year_index)


def WeekOfYearSeasonalISSM():
    return SeasonalityISSM(num_seasons=53, time_feature=week_of_year_index)


def DayOfWeekSeasonalISSM():
    return SeasonalityISSM(num_seasons=7, time_feature=day_of_week_index)


def HourOfDaySeasonalISSM():
    return SeasonalityISSM(num_seasons=24, time_feature=hour_of_day_index)


def MinuteOfHourSeasonalISSM():
    return SeasonalityISSM(num_seasons=60, time_feature=minute_of_hour_index)


class CompositeISSM(ISSM):
    DEFAULT_ADD_TREND: bool = True

    @validated()
    def __init__(
        self,
        seasonal_issms: List[SeasonalityISSM],
        add_trend: bool = DEFAULT_ADD_TREND,
    ) -> None:
        super().__init__()
        self.seasonal_issms = seasonal_issms
        self.nonseasonal_issm = (
            LevelISSM() if add_trend is False else LevelTrendISSM()
        )

    def latent_dim(self) -> int:
        return (
            sum(issm.latent_dim() for issm in self.seasonal_issms)
            + self.nonseasonal_issm.latent_dim()
        )

    def output_dim(self) -> int:
        return self.nonseasonal_issm.output_dim()

    def time_features(self) -> List[TimeFeature]:
        ans = self.nonseasonal_issm.time_features()
        for issm in self.seasonal_issms:
            ans.extend(issm.time_features())
        return ans

    @classmethod
    def get_from_freq(cls, freq: str, add_trend: bool = DEFAULT_ADD_TREND):
        offset = to_offset(freq)

        seasonal_issms: List[SeasonalityISSM] = []

        if offset.name in ["M", "ME"]:
            seasonal_issms = [MonthOfYearSeasonalISSM()]
        elif norm_freq_str(offset.name) == "W":
            seasonal_issms = [WeekOfYearSeasonalISSM()]
        elif offset.name == "D":
            seasonal_issms = [DayOfWeekSeasonalISSM()]
        elif offset.name == "B":  # TODO: check this case
            seasonal_issms = [DayOfWeekSeasonalISSM()]
        elif offset.name in ["H", "h"]:
            seasonal_issms = [
                HourOfDaySeasonalISSM(),
                DayOfWeekSeasonalISSM(),
            ]
        elif offset.name in ["T", "min"]:
            seasonal_issms = [
                MinuteOfHourSeasonalISSM(),
                HourOfDaySeasonalISSM(),
            ]
        else:
            RuntimeError(f"Unsupported frequency {offset.name}")

        return cls(seasonal_issms=seasonal_issms, add_trend=add_trend)

