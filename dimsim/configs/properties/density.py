import typing

from pydantic import Field

from ..workflow import PhaseConfig
from .base import PropertyConfig


class DensityConfig(PropertyConfig):
    """
    Configuration for the Density property calculation.
    """

    name: typing.ClassVar[typing.Literal["Density"]] = "Density"

    bulk: PhaseConfig = Field(
        default_factory=PhaseConfig,
        description="Workflow configuration for bulk density calculations."
    )
