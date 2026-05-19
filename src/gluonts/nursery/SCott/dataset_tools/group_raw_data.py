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

from pts.dataset.repository.datasets import get_dataset, dataset_recipes
from pts.dataset.utils import to_pandas
import numpy as np
import pandas as pd
from pts.dataset import ListDataset
import torch
from pts.model.simple_feedforward import SimpleFeedForwardEstimator
import pickle
import json
import random




























