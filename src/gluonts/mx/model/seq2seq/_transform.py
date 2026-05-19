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
from numpy.lib.stride_tricks import as_strided

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import FlatMapTransformation


class ForkingSequenceSplitter(FlatMapTransformation):
    """
    Forking sequence splitter.
    """

    @validated()
    def __init__(
        self,
        instance_sampler,
        enc_len: int,
        dec_len: int,
        num_forking: Optional[int] = None,
        target_field: str = FieldName.TARGET,
        encoder_series_fields: Optional[List[str]] = None,
        decoder_series_fields: Optional[List[str]] = None,
        encoder_disabled_fields: Optional[List[str]] = None,
        decoder_disabled_fields: Optional[List[str]] = None,
        prediction_time_decoder_exclude: Optional[List[str]] = None,
        is_pad_out: str = "is_pad",
        start_input_field: str = "start",
    ) -> None:
        super().__init__()

        assert enc_len > 0, "The value of `enc_len` should be > 0"
        assert dec_len > 0, "The value of `dec_len` should be > 0"

        self.instance_sampler = instance_sampler
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.num_forking = (
            num_forking if num_forking is not None else self.enc_len
        )
        self.target_field = target_field

        self.encoder_series_fields = (
            encoder_series_fields + [self.target_field]
            if encoder_series_fields is not None
            else [self.target_field]
        )
        self.decoder_series_fields = (
            decoder_series_fields + [self.target_field]
            if decoder_series_fields is not None
            else [self.target_field]
        )

        self.encoder_disabled_fields = (
            encoder_disabled_fields
            if encoder_disabled_fields is not None
            else []
        )

        self.decoder_disabled_fields = (
            decoder_disabled_fields
            if decoder_disabled_fields is not None
            else []
        )

        # Fields that are not used at prediction time for the decoder
        self.prediction_time_decoder_exclude = (
            prediction_time_decoder_exclude + [self.target_field]
            if prediction_time_decoder_exclude is not None
            else [self.target_field]
        )

        self.is_pad_out = is_pad_out
        self.start_in = start_input_field



