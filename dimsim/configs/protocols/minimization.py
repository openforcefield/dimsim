import typing

import pydantic
from openff.units import unit

from ...utils.pydantic import OpenFFQuantity
from .base import ProtocolConfig


class MinimizationConfig(ProtocolConfig):
    """
    Configuration for the Minimization protocol.
    """

    name: typing.ClassVar[str] = "Minimization"

    max_iterations: int = pydantic.Field(
        default=1000, description="Maximum number of minimization iterations"
    )
    tolerance: OpenFFQuantity[unit.kilojoule / unit.mole] = pydantic.Field(
        default=10.0 * unit.kilojoule / unit.mole,
        description="Energy tolerance for minimization in kJ/mol"
    )
