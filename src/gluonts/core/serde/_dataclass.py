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

import dataclasses
import inspect
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    cast,
    Any,
    Callable,
    ClassVar,
    Generic,
    TypeVar,
)

from gluonts.itertools import select
from gluonts.pydantic import pydantic, dataclass as pydantic_dataclass

T = TypeVar("T")


class _EventualType:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)

        return cls._instance

    def __repr__(self):
        return "EVENTUAL"


EVENTUAL = cast(Any, _EventualType())


@dataclasses.dataclass
class Eventual(Generic[T]):
    value: T



    def unwrap(self) -> T:
        if self.value is EVENTUAL:
            raise ValueError("UNWRAP")

        return self.value


# OrElse should have `Any` type, meaning that we can do
#     x: int = OrElse(...)
# without mypy complaining. We achieve this by nominally deriving from Any
# during type checking. Because it's not actually possible to derive from Any
# in Python, we inherit from object during runtime instead.
if TYPE_CHECKING:
    AnyLike = Any
else:
    AnyLike = object


@dataclasses.dataclass
class OrElse(AnyLike):
    """
    A default field for a dataclass, that uses a function to calculate the
    value if none was passed.

    The function can take arguments, which are other fields of the annotated
    dataclass, which can also be `OrElse` fields. In case of circular
    dependencies an error is thrown.

    ::

        @serde.dataclass
        class X:
            a: int
            b: int = serde.OrElse(lambda a: a + 1)
            c: int = serde.OrEsle(lambda c: c * 2)

        x = X(1)
        assert x.b == 2
        assert x.c == 4
    ```
    """

    fn: Callable
    parameters: dict = dataclasses.field(init=False)

    def __post_init__(self):
        self.parameters = set(inspect.signature(self.fn).parameters)



def dataclass(
    cls=None,
    *,
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
):
    """
    Custom dataclass wrapper for serde.

    This works similar to the ``dataclasses.dataclass`` and
    ``pydantic.dataclasses.dataclass`` decorators. Similar to the ``pydantic``
    version, this does type checking of arguments during runtime.
    """
    pass




if TYPE_CHECKING:
    dataclass = dataclasses.dataclass  # type: ignore # noqa
