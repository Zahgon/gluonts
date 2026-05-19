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

import functools
import gzip
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast, Optional, BinaryIO, List

import numpy as np
import pandas as pd
from toolz import first, take, valmap


from gluonts import maybe
from gluonts import json
from gluonts.exceptions import GluonTSDataError

from . import Dataset, DatasetWriter


def load(file_obj):
    return map(json.loads, file_obj)


def dump(objects, file_obj):
    for object_ in objects:
        json.dump(object_, file_obj, nl=True)


@functools.singledispatch
def encode_json(arg):
    if isinstance(arg, (str, int)):
        return arg

    if isinstance(arg, float):
        if np.isnan(arg):
            return "NaN"
        elif np.isposinf(arg):
            return "Infinity"
        elif np.isneginf(arg):
            return "-Infinity"
        return arg

    if isinstance(arg, datetime):
        return str(arg)

    raise ValueError(f"Can't encode {arg!r}")










@dataclass(frozen=True)
class JsonLinesFile:
    """
    An iterable type that draws from a JSON Lines file.

    Parameters
    ----------
    path
        Path of the file to load data from. This should be a valid
        JSON Lines file.
    """

    SUFFIXES = {
        ".json",
        ".json.gz",
        ".jsonl",
        ".jsonl.gz",
    }

    path: Path
    start: int = 0
    n: Optional[int] = None

    line_starts: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.line_starts:
            self.line_starts.extend(self._line_starts())


    def __iter__(self):
        with self.open() as jsonl_file:
            jsonl_file.seek(self.line_starts[self.start])

            for line_number, line in take(
                self.n,
                enumerate(jsonl_file, start=self.start),
            ):
                try:
                    yield json.loads(line)
                except ValueError:
                    raise GluonTSDataError(
                        f"Could not read json line {line_number}, {line}"
                    )

    def __len__(self):
        return maybe.box(self.n).unwrap_or(len(self.line_starts) - self.start)

    def _line_starts(self):
        """
        Calculate the position for each line in the file.

        This information can be used with ``file.seek`` to directly jump to a
        specific line in the file.
        """
        pass

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # TODO: should we enable steps other than 1?
            assert idx.step is None or idx.step == 1

            # normalize index
            start, stop, _step = idx.indices(len(self))
            idx = slice(start + self.start, stop + self.start)

            line_starts = self.line_starts[idx]
            if len(line_starts) == 0:
                return JsonLinesFile(
                    self.path, n=0, line_starts=self.line_starts
                )

            return JsonLinesFile(
                self.path,
                start=idx.start,
                n=len(line_starts),
                line_starts=self.line_starts,
            )

        return first(self[idx:])


@dataclass
class JsonLinesWriter(DatasetWriter):
    use_gzip: bool = True
    suffix: str = ".json"
    # Python uses `compresslevel=9` by default, which is very slow
    # We opt for faster writes by default, for more modest size savings
    compresslevel: int = 4

    def write_to_file(self, dataset: Dataset, path: Path) -> None:
        if self.use_gzip:
            out_file = cast(
                BinaryIO,
                gzip.open(path, "wb", compresslevel=self.compresslevel),
            )
        else:
            out_file = open(path, "wb")

        with out_file:
            for entry in dataset:
                json.bdump(encode_json(entry), out_file, nl=True)

    def write_to_folder(
        self, dataset: Dataset, folder: Path, name: Optional[str] = None
    ) -> None:
        if name is None:
            name = "data"

        if self.use_gzip:
            suffix = self.suffix + ".gz"
        else:
            suffix = self.suffix

        self.write_to_file(dataset, (folder / name).with_suffix(suffix))
