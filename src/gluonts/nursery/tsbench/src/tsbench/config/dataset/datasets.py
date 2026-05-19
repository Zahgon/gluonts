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

# pylint: disable=too-many-lines
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Dict, List, Tuple
import numpy as np
import pandas as pd
from ._factory import register_dataset
from .preprocessing import ConstantTargetFilter, Filter
from .sources import (
    GluonTsDatasetConfig,
    KaggleDatasetConfig,
    M3DatasetConfig,
    MonashDatasetConfig,
)


@register_dataset
@dataclass(frozen=True)
class ExchangeRateDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `exchange_rate_nips` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "exchange_rate"




@register_dataset
@dataclass(frozen=True)
class ElectricityDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `electricity_nips` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "electricity"



@register_dataset
@dataclass(frozen=True)
class SolarDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `solar_nips` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "solar"




@register_dataset
@dataclass(frozen=True)
class WikiDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `wiki-rolling_nips` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "wiki"




@register_dataset
@dataclass(frozen=True)
class TaxiDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `taxi_30min` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "taxi"




@register_dataset
@dataclass(frozen=True)
class M3MonthlyDatasetConfig(M3DatasetConfig):
    """
    The dataset configuration for the `m3_monthly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m3_monthly"



@register_dataset
@dataclass(frozen=True)
class M3QuarterlyDatasetConfig(M3DatasetConfig):
    """
    The dataset configuration for the `m3_quarterly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m3_quarterly"



@register_dataset
@dataclass(frozen=True)
class M3YearlyDatasetConfig(M3DatasetConfig):
    """
    The dataset configuration for the `m3_yearly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m3_yearly"



@register_dataset
@dataclass(frozen=True)
class M3OtherDatasetConfig(M3DatasetConfig):
    """
    The dataset configuration for the `m3_other` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m3_other"




@register_dataset
@dataclass(frozen=True)
class M4HourlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m4_hourly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m4_hourly"




@register_dataset
@dataclass(frozen=True)
class M4DailyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m4_daily` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m4_daily"




@register_dataset
@dataclass(frozen=True)
class M4WeeklyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m4_weekly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m4_weekly"




@register_dataset
@dataclass(frozen=True)
class M4MonthlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m4_monthly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m4_monthly"




@register_dataset
@dataclass(frozen=True)
class M4QuarterlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m4_quarterly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m4_quarterly"




@register_dataset
@dataclass(frozen=True)
class M4YearlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m4_yearly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m4_yearly"




@register_dataset
@dataclass(frozen=True)
class M5DatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `m5` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "m5"


    def _materialize(self, directory: Path, regenerate: bool = False) -> None:
        shutil.copytree(
            Path.home() / ".mxnet" / "gluon-ts" / "datasets" / "m5",
            directory / "m5",
        )
        super()._materialize(directory, regenerate=True)


@register_dataset
@dataclass(frozen=True)
class TourismMonthlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `tourism_monthly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "tourism_monthly"



@register_dataset
@dataclass(frozen=True)
class TourismQuarterlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `tourism_quarterly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "tourism_quarterly"



@register_dataset
@dataclass(frozen=True)
class TourismYearlyDatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `tourism_yearly` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "tourism_yearly"



@register_dataset
@dataclass(frozen=True)
class NN5DatasetConfig(GluonTsDatasetConfig):
    """
    The dataset configuration for the `nn5_daily_without_missing` dataset.
    """

    @classmethod
    def name(cls) -> str:
        return "nn5"




@register_dataset
@dataclass(frozen=True)
class LondonSmartMetersDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "London Smart Meters".
    """

    @classmethod
    def name(cls) -> str:
        return "london_smart_meters"






@register_dataset
@dataclass(frozen=True)
class WindFarmsDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Wind Farms".
    """

    @classmethod
    def name(cls) -> str:
        return "wind_farms"


    def _filters(self, prediction_length: int) -> List[Filter]:
        return [
            ConstantTargetFilter(prediction_length, required_length=100000)
        ]





@register_dataset
@dataclass(frozen=True)
class CarPartsDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Car Parts".
    """

    @classmethod
    def name(cls) -> str:
        return "car_parts"






@register_dataset
@dataclass(frozen=True)
class DominickDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Dominick".
    """

    @classmethod
    def name(cls) -> str:
        return "dominick"








@register_dataset
@dataclass(frozen=True)
class FredMdDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Federal Reserve Economic Dataset".
    """

    @classmethod
    def name(cls) -> str:
        return "fred_md"






@register_dataset
@dataclass(frozen=True)
class SanFranciscoTrafficDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "San Francisco Traffic".
    """

    @classmethod
    def name(cls) -> str:
        return "san_francisco_traffic"






@register_dataset
@dataclass(frozen=True)
class PedestrianCountDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Pedestrian Counts".
    """

    @classmethod
    def name(cls) -> str:
        return "pedestrian_count"






@register_dataset
@dataclass(frozen=True)
class HospitalDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Hospitals".
    """

    @classmethod
    def name(cls) -> str:
        return "hospital"






@register_dataset
@dataclass(frozen=True)
class CovidDeathsDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "COVID Deaths".
    """

    @classmethod
    def name(cls) -> str:
        return "covid_deaths"






@register_dataset
@dataclass(frozen=True)
class KddCupDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "KDD Cup 2018".
    """

    @classmethod
    def name(cls) -> str:
        return "kdd_2018"






@register_dataset
@dataclass(frozen=True)
class CifDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "CIF 2016".
    """

    @classmethod
    def name(cls) -> str:
        return "cif_2016"







@register_dataset
@dataclass(frozen=True)
class AustralianElectricityDemandDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Australian Electricity Demand".
    """

    @classmethod
    def name(cls) -> str:
        return "australian_electricity_demand"






@register_dataset
@dataclass(frozen=True)
class BitcoinDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Bitcoin".
    """

    @classmethod
    def name(cls) -> str:
        return "bitcoin"






@register_dataset
@dataclass(frozen=True)
class RideshareDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Rideshare".
    """

    @classmethod
    def name(cls) -> str:
        return "rideshare"






@register_dataset
@dataclass(frozen=True)
class VehicleTripsDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Vehicle Trips".
    """

    @classmethod
    def name(cls) -> str:
        return "vehicle_trips"







@register_dataset
@dataclass(frozen=True)
class WeatherDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Weather".
    """

    @classmethod
    def name(cls) -> str:
        return "weather"






@register_dataset
@dataclass(frozen=True)
class TemperatureRainDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "Temperature Rain".
    """

    @classmethod
    def name(cls) -> str:
        return "temperature_rain"






@register_dataset
@dataclass(frozen=True)
class M1YearlyDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "M1 Yearly".
    """

    @classmethod
    def name(cls) -> str:
        return "m1_yearly"






@register_dataset
@dataclass(frozen=True)
class M1QuarterlyDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "M1 Quarterly".
    """

    @classmethod
    def name(cls) -> str:
        return "m1_quarterly"






@register_dataset
@dataclass(frozen=True)
class M1MonthlyDatasetConfig(MonashDatasetConfig):
    """
    The dataset configuration for "M1 Monthly".
    """

    @classmethod
    def name(cls) -> str:
        return "m1_monthly"






@register_dataset
@dataclass(frozen=True)
class RossmannDatasetConfig(KaggleDatasetConfig):
    """
    The dataset configuration for the "Rossmann Store Sales" Kaggle
    competition.
    """

    @classmethod
    def name(cls) -> str:
        return "rossmann"



    def _extract_data(
        self, path: Path
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Read the raw data
        data = cast(pd.DataFrame, pd.read_csv(path / "train.csv"))
        stores = cast(pd.DataFrame, pd.read_csv(path / "store.csv"))

        # Generate GluonTS dataset
        metadata = {
            "freq": "D",
            "prediction_length": 48,
            "feat_static_cat": [
                {
                    "name": "store",
                    "cardinality": len(stores),
                },
            ],
        }

        series = []
        for i, store_data in data.groupby("Store"):
            sorted_data = store_data.sort_values("Date")
            series.append(
                {
                    "item_id": int(i) - 1,
                    "start": sorted_data.Date.min(),
                    "target": sorted_data.Sales.to_list(),
                    "feat_static_cat": [
                        int(i) - 1,
                    ],
                }
            )

        return metadata, series


@register_dataset
@dataclass(frozen=True)
class CorporacionFavoritaDatasetConfig(KaggleDatasetConfig):
    """
    The dataset configuration for the "Corporación Favorita" Kaggle
    competition.
    """

    @classmethod
    def name(cls) -> str:
        return "corporacion_favorita"



    def _extract_data(
        self, path: Path
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Read the raw data
        data = cast(pd.DataFrame, pd.read_csv(path / "train.csv"))
        stores = cast(pd.DataFrame, pd.read_csv(path / "stores.csv"))
        item_ids = np.sort(data.item_nbr.unique())

        # Generate GluonTS dataset
        metadata = {
            "freq": "D",
            "prediction_length": 16,
            "feat_static_cat": [
                {
                    "name": "store",
                    "cardinality": len(stores),
                },
                {
                    "name": "item",
                    "cardinality": len(item_ids),
                },
            ],
        }

        series = []
        for i, ((item, store_id), group_data) in enumerate(
            data.groupby(["item_nbr", "store_nbr"])
        ):
            item_id = np.where(item_ids == item)[0][0]
            sorted_data = group_data.sort_values("date")
            sales = pd.Series(
                sorted_data.unit_sales.to_numpy(),
                index=pd.DatetimeIndex(sorted_data.date),
            )
            series.append(
                {
                    "item_id": i,
                    "start": sorted_data.date.min(),
                    "target": sales.resample("D")
                    .first()
                    .fillna(value=0)
                    .to_list(),
                    "feat_static_cat": [
                        int(store_id) - 1,
                        int(item_id),
                    ],
                }
            )

        return metadata, series


@register_dataset
@dataclass(frozen=True)
class WalmartDatasetConfig(KaggleDatasetConfig):
    """
    The dataset configuration for the "Walmart Recruiting" Kaggle competition.
    """

    @classmethod
    def name(cls) -> str:
        return "walmart"



    def _extract_data(
        self, path: Path
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Read the raw data
        data = cast(pd.DataFrame, pd.read_csv(path / "train.csv"))
        department_ids = np.sort(data.Dept.unique())

        # Generate GluonTS dataset
        metadata = {
            "freq": "W",
            "prediction_length": 39,
            "feat_static_cat": [
                {
                    "name": "store",
                    "cardinality": len(
                        data.Store.unique()
                    ),  # pylint: disable=no-member
                },
                {
                    "name": "department",
                    "cardinality": len(department_ids),
                },
            ],
        }

        series = []
        # pylint: disable=no-member
        for i, ((store_id, department), group_data) in enumerate(
            data.groupby(["Store", "Dept"])
        ):
            department_id = np.where(department_ids == department)[0][0]
            sorted_data = group_data.sort_values("Date")
            series.append(
                {
                    "item_id": i,
                    "start": sorted_data.Date.min(),
                    "target": sorted_data.Weekly_Sales.to_list(),
                    "feat_static_cat": [
                        int(store_id) - 1,
                        int(department_id),
                    ],
                }
            )

        return metadata, series


@register_dataset
@dataclass(frozen=True)
class RestaurantDatasetConfig(KaggleDatasetConfig):
    """
    The dataset configuration for the "Restaurant" Kaggle competition.
    """

    @classmethod
    def name(cls) -> str:
        return "restaurant"



    def _extract_data(
        self, path: Path
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Read the raw data
        data = cast(pd.DataFrame, pd.read_csv(path / "air_visit_data.csv"))
        store_ids = np.sort(data.air_store_id.unique())

        # Generate GluonTS dataset
        metadata = {
            "freq": "D",
            "prediction_length": 39,
            "feat_static_cat": [
                {
                    "name": "restaurant",
                    "cardinality": len(store_ids),
                },
            ],
        }

        series = []
        # pylint: disable=no-member
        for i, (store, group_data) in enumerate(data.groupby("air_store_id")):
            store_id = np.where(store_ids == store)[0][0]
            sorted_data = group_data.sort_values("visit_date")
            visitors = pd.Series(
                sorted_data.visitors.to_numpy(),
                index=pd.DatetimeIndex(sorted_data.visit_date),
            )
            series.append(
                {
                    "item_id": i,
                    "start": sorted_data.visit_date.min(),
                    "target": visitors.resample("D")
                    .first()
                    .fillna(value=0)
                    .to_list(),
                    "feat_static_cat": [
                        int(store_id),
                    ],
                }
            )

        return metadata, series
