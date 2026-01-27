import typing

import pydantic
from openff.units import Quantity, unit

from ...utils.pydantic import OpenFFQuantity
from .base import ProtocolConfig


class InitialEquilibrationConfig(ProtocolConfig):
    """
    Configuration for the InitialEquilibration protocol.
    """

    name: typing.ClassVar[typing.Literal["InitialEquilibration"]] = "InitialEquilibration"

    timestep: OpenFFQuantity[unit.femtosecond] = pydantic.Field(
        default=2.0 * unit.femtosecond,
        description="Integration timestep in femtoseconds."
    )

    n_steps: int = pydantic.Field(
        1_000_000,
        description="Number of integration steps for initial equilibration."
    )

    ensemble: typing.Literal["NVT", "NPT"] = pydantic.Field(
        ...,
        description="Ensemble to use during initial equilibration."
    )

    friction_coefficient: OpenFFQuantity[1 / unit.picosecond] = pydantic.Field(
        default=1.0 / unit.picosecond,
        description="Friction coefficient for the Langevin integrator in 1/ps."
    )

    @property
    def length(self) -> Quantity:
        """Base length of the Equilibration step in nanoseconds."""
        return (self.n_steps * self.timestep).to(unit.nanoseconds)


class EquilibrationConfig(InitialEquilibrationConfig):
    """
    Configuration for the Equilibration protocol.
    """

    name: typing.ClassVar[typing.Literal["Equilibration"]] = "Equilibration"

    condition_aggregation: typing.Literal["any", "all"] = pydantic.Field(
        default="all",
        description=(
            "Method to aggregate multiple sampling conditions. "
            "'all' requires all conditions to be met, 'any' requires any one condition to be met."
        )
    )

    max_iterations: int = pydantic.Field(
        default=10,
        description="Maximum number of Equilibration iterations."
    )

    n_uncorrelated_samples: int = pydantic.Field(
        50,
        description="Number of uncorrelated samples to collect to determine Equilibration."
    )

    observables: list[
        typing.Literal[
            "temperature", "volume", "density", "potential_energy",
            "kinetic_energy", "total_energy"
        ]] = pydantic.Field(
        ...,
        description="Observable to monitor for Equilibration."
    )

    store_equilibrated_coordinates: bool = pydantic.Field(
        default=True,
        description=(
            "Whether to store the equilibrated coordinates in the coordinate store. "
            "If True, the coordinates will be stored with force_field_id 'equilibration'. "
        ),
    )

    store_lowest_energy_replicate_only: bool = pydantic.Field(
        default=False,
        description=(
            "If True, only the equilibrated coordinates from the replicate "
            "with the lowest potential energy will be stored. "
            "If False, all replicates will be stored."
        )
    )

    @property
    def max_length(self) -> Quantity:
        """Maximum length of the Equilibration in nanoseconds."""
        return self.max_iterations * self.length
