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
import inspect
import logging
from collections import OrderedDict
from functools import singledispatch
from typing import Any, Type, TypeVar

import numpy as np

from gluonts.core import fqname_for
from gluonts.exceptions import GluonTSHyperparametersError
from gluonts.pydantic import (
    BaseConfig,
    BaseModel,
    ValidationError,
    create_model,
)


logger = logging.getLogger(__name__)

A = TypeVar("A")


def from_hyperparameters(cls: Type[A], **hyperparameters) -> A:
    """
    Reflectively create an instance of a class with a :func:`validated`
    initializer.

    Parameters
    ----------
    cls
        The type ``A`` of the component to be instantiated.
    hyperparameters
        A dictionary of key-value pairs to be used as parameters to the
        component initializer.

    Returns
    -------
    A
        An instance of the given class.

    Raises
    ------
    GluonTSHyperparametersError
        Wraps a :class:`ValidationError` thrown when validating the
        initializer parameters.
    """
    Model = getattr(cls.__init__, "Model", None)

    try:
        if Model is not None:
            return cls(**Model(**hyperparameters).__dict__)  # type: ignore
        else:
            return cls(**hyperparameters)  # type: ignore
    except ValidationError as error:
        raise GluonTSHyperparametersError from error


@singledispatch
def equals(this: Any, that: Any) -> bool:
    """
    Structural equality check between two objects of arbitrary type.

    By default, this function delegates to :func:`equals_default_impl`.

    In addition, the function dispatches to specialized implementations based
    on the type of the first argument, so the above conditions might be
    stricter for certain types.

    Parameters
    ----------
    this, that
        Objects to compare.

    Returns
    -------
    bool
        A boolean value indicating whether ``this`` and ``that`` are
        structurally equal.

    See Also
    --------
    equals_default_impl
        Default semantics of a structural equality check between two objects
        of arbitrary type.
    equals_representable_block
        Specialization for Gluon :class:`~mxnet.gluon.HybridBlock` input
        arguments.
    equals_parameter_dict
        Specialization for Gluon :class:`~mxnet.gluon.ParameterDict` input
        arguments.
    """
    return equals_default_impl(this, that)


def equals_default_impl(this: Any, that: Any) -> bool:
    """
    Default semantics of a structural equality check between two objects of
    arbitrary type.

    Two objects ``this`` and ``that`` are defined to be structurally equal
    if and only if the following criteria are satisfied:

    1. Their types match.
    2. If their initializer are :func:`validated`, their initializer arguments
       are pairwise structurally equal.
    3. If their initializer are not :func:`validated`, they are referentially
       equal (i.e. ``this == that``).

    Parameters
    ----------
    this, that
        Objects to compare.

    Returns
    -------
    bool
        A boolean value indicating whether ``this`` and ``that`` are
        structurally equal.
    """
    if type(this) != type(that):
        return False

    if hasattr(this, "__init_args__") and hasattr(that, "__init_args__"):
        return equals(
            this.__init_args__,
            that.__init_args__,
        )

    if hasattr(this, "__init_passed_kwargs__") and hasattr(
        that, "__init_passed_kwargs__"
    ):
        return equals(
            this.__init_passed_kwargs__,
            that.__init_passed_kwargs__,
        )

    return this == that


@equals.register(list)
def equals_list(this: list, that: list) -> bool:
    if not len(this) == len(that):
        return False

    for x, y in zip(this, that):
        if not equals(x, y):
            return False

    return True






@singledispatch
def tensor_to_numpy(tensor) -> np.ndarray:
    raise NotImplementedError




@singledispatch
def skip_encoding(v: Any) -> bool:
    """
    Tells whether the input value `v` should be encoded using the
    :func:`~gluonts.core.serde.encode` function.

    This is used by :func:`validated` to determine which values need to
    be skipped when recording the initializer arguments for later
    serialization.

    This is the fallback implementation, and can be specialized for
    specific types by registering handler functions.
    """
    return False


class BaseValidatedInitializerModel(BaseModel):
    """
    Base Pydantic model for components with :func:`validated` initializers.

    See Also
    --------
    validated
        Decorates an initializer methods with argument validation logic.
    """

    class Config(BaseConfig):
        """
        `Config <https://pydantic-docs.helpmanual.io/#model-config>`_ for the
        Pydantic model inherited by all :func:`validated` initializers.

        Allows the use of arbitrary type annotations in initializer parameters.
        """

        arbitrary_types_allowed = True


def validated(base_model=None):
    """
    Decorates an ``__init__`` method with typed parameters with validation and
    auto-conversion logic.

    >>> class ComplexNumber:
    ...     @validated()
    ...     def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
    ...         self.x = x
    ...         self.y = y

    Classes with decorated initializers can be instantiated using arguments of
    another type (e.g. an ``y`` argument of type ``str`` ). The decorator
    handles the type conversion logic.

    >>> c = ComplexNumber(y='42')
    >>> (c.x, c.y)
    (0.0, 42.0)

    If the bound argument cannot be converted, the decorator throws an error.

    >>> c = ComplexNumber(y=None)
    Traceback (most recent call last):
        ...
    pydantic.v1.error_wrappers.ValidationError: 1 validation error for
    ComplexNumberModel
    y
      none is not an allowed value (type=type_error.none.not_allowed)

    Internally, the decorator delegates all validation and conversion logic to
    `a Pydantic model <https://pydantic-docs.helpmanual.io/>`_, which can be
    accessed through the ``Model`` attribute of the decorated initializer.

    >>> ComplexNumber.__init__.Model
    <class 'pydantic.v1.main.ComplexNumberModel'>

    The Pydantic model is synthesized automatically from on the parameter
    names and types of the decorated initializer. In the ``ComplexNumber``
    example, the synthesized Pydantic model corresponds to the following
    definition.

    >>> class ComplexNumberModel(BaseValidatedInitializerModel):
    ...     x: float = 0.0
    ...     y: float = 0.0


    Clients can optionally customize the base class of the synthesized
    Pydantic model using the ``base_model`` decorator parameter. The default
    behavior uses :class:`BaseValidatedInitializerModel` and its
    `model config <https://pydantic-docs.helpmanual.io/#config>`_.

    See Also
    --------
    BaseValidatedInitializerModel
        Default base class for all synthesized Pydantic models.
    """


    return validator
