import typing

from pydantic import Field

from ..protocols import (
    CoordinateGenerationConfig,
    EquilibrationConfig,
    InitialEquilibrationConfig,
    SimulationConfig,
)
from ..workflow import PhaseConfig
from .base import PropertyConfig


def _generate_vacuum_config() -> PhaseConfig:
    """Generate a config suitable for a vacuum simulation"""
    return PhaseConfig(
        coordinate_generation=CoordinateGenerationConfig(
            n_max_molecules=1,
        ),
        initial_equilibration=InitialEquilibrationConfig(
            ensemble="NVT",
        ),
        equilibration=EquilibrationConfig(
            ensemble="NVT",
        ),
        simulation=SimulationConfig(
            ensemble="NVT",
        ),
    )

class EnthalpyOfVaporizationConfig(PropertyConfig):
    """
    Configuration for the Enthalpy of Vaporization property calculation.
    """

    name: typing.ClassVar[typing.Literal["dhvap"]] = "dhvap"

    bulk: PhaseConfig = Field(
        default_factory=PhaseConfig,
        description=(
            "Workflow configuration for bulk enthalpy of vaporization "
            "calculations."
        )
    )

    vacuum: PhaseConfig = Field(
        default_factory=_generate_vacuum_config,
        description=(
            "Workflow configuration for vacuum enthalpy of vaporization "
            "calculations."
        )
    )
