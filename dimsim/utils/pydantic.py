"""Pydantic utilities for custom type handling

Heavily borrowed from Simon Boothroyd's pydantic_units project for OpenMM.
"""

import functools
import typing

import openff.units
import pint
import pydantic
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema
from pydantic_core.core_schema import (
    json_or_python_schema,
    no_info_plain_validator_function,
    plain_serializer_function_ser_schema,
    str_schema,
)

if typing.TYPE_CHECKING:
    import openmm.unit


class _OpenFFQuantityAnnotation:
    """Pydantic annotation for openff.units.Quantity objects.

    This annotation only handles serialization, as validation is performed by
    the BeforeValidator in the metaclass __getitem__ method.

    Examples
    --------

    .. code-block:: python

        from dimsim.utils.pydantic import OpenFFQuantity

        class MyModel(BaseModel):
            # or simply use the pre-defined Quantity type:
            # temperature: OpenFFQuantity[openff.units.unit.kelvin]
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: typing.Any,
        handler: pydantic.GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Define serialization for Quantity objects.

        Validation is handled by _quantity_validator via BeforeValidator,
        so this only needs to handle serialization.
        """

        def serialize_quantity(value: openff.units.Quantity) -> str:
            """Convert Quantity to string."""
            return str(value)

        # Use the default handler for validation (passes through the Quantity)
        python_schema = no_info_plain_validator_function(lambda x: x)

        return json_or_python_schema(
            json_schema=python_schema,
            python_schema=python_schema,
            serialization=plain_serializer_function_ser_schema(
                serialize_quantity,
                return_schema=str_schema(),
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: pydantic.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Define JSON schema representation."""
        return handler(str_schema())


def _quantity_validator(
    value: openff.units.Quantity | str | "openmm.unit.Quantity",
    expected_units: openff.units.Unit,
) -> openff.units.Quantity:
    """Validate that a value is an openff.units.Quantity with expected units."""
    if isinstance(value, str):
        quantity = openff.units.Quantity(value)

    elif isinstance(value, openff.units.Quantity):
        quantity = value

    elif 'openmm.unit' in str(type(value)):

        from openff.units.openmm import to_openff_quantity
        quantity = to_openff_quantity(value)

    else:
        raise TypeError(f"Cannot convert type {type(value)} to openff.units.Quantity")

    try:
        return quantity.to(expected_units)
    except pint.errors.DimensionalityError as e:
        raise ValueError(
            f"Quantity '{quantity}' does not have expected units of '{expected_units}'"
        ) from e


class _OpenFFQuantityMeta(type):
    """Metaclass to validate unit dimensions"""
    def __getitem__(cls, item: openff.units.Unit):
        validator = functools.partial(_quantity_validator, expected_units=item)

        return typing.Annotated[
            openff.units.Quantity, _OpenFFQuantityAnnotation, pydantic.BeforeValidator(validator)
        ]


class OpenFFQuantity(metaclass=_OpenFFQuantityMeta):
    """A pydantic-compatible Quantity type with unit dimension validation."""
