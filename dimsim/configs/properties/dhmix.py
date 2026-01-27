import typing

from pydantic import Field

from ..workflow import PhaseConfig
from .base import PropertyConfig


class EnthalpyOfMixingConfig(PropertyConfig):
    """
    Configuration for the dhmix property calculation.
    """

    name: typing.ClassVar[typing.Literal["dhmix"]] = "dhmix"

    bulk: PhaseConfig = Field(
        default_factory=PhaseConfig,
        description="Workflow configuration for bulk enthalpy of mixing calculations."
    )

    pure: PhaseConfig = Field(
        default_factory=PhaseConfig,
        description="Workflow configuration for pure component enthalpy calculations."
    )
