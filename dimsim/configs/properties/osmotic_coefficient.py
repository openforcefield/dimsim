import typing

import numpy as np
from openff.interchange.components._packmol import UNIT_CUBE
from pydantic import Field

from ..protocols import (
    AddForceConfig,
    CoordinateGenerationConfig,
    EquilibrationConfig,
    GenerateSystemConfig,
    InitialEquilibrationConfig,
    SimulationConfig,
)
from ..workflow import PhaseConfig
from .base import PropertyConfig


def _generate_osmotic_coefficient_config() -> PhaseConfig:
    """
    Generate sensible defaults for osmotic pressure calculations.

    # TODO: come up with better defaults here
    # key is to have a big enough box along the force axis
    # to accommodate the flat bottom restraint
    # unsure if we can do this well without a 'concentration' argument
    """
    axis = OsmoticCoefficientConfig.force_axis
    box = np.array(UNIT_CUBE) * 0.5
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    box[axis_index] *= 2.0  # elongate box along force axis

    return PhaseConfig(
        coordinate_generation=CoordinateGenerationConfig(
            n_max_molecules=4000, # this needs to be big enough for the ion box
            box_shape=box,
        ),
        system_generation=GenerateSystemConfig(
            additional_forces={
                OsmoticCoefficientConfig.force_name:  AddForceConfig(
                    expression=f"0.5*k*max(0, abs({axis}-{axis}0)-rfb)^2",
                    global_parameters={
                        "k": "4184 kJ/(nm**2 mol)",
                        "rfb": "1 nm",
                        f"{axis}0": "3 nm",
                    },
                    smarts_patterns=["[!+0:1]"],
                    # TODO: necessary for multi-atom charged molecules like NH4+ ?
                    whole_molecule=True,
                    force_type="CustomExternalForce",
                ),
            },
        ),

        # equilibrate NPT to get the density about right
        # assume water equilibrates reasonably quickly
        initial_equilibration=InitialEquilibrationConfig(
            n_steps=100_000, # 100_000 * 0.002 ps = 200 ps
            ensemble="NPT",
        ),
        equilibration=EquilibrationConfig(
            n_steps=1_000_000, # 1_000_000 * 0.002 ps = 2 ns
            ensemble="NVT",
            observables=["potential_energy"],
        ),
        simulation=SimulationConfig(
            # TODO: this seems long
            n_steps=10_000_000, # 10_000_000 * 0.002 ps = 20 ns
            ensemble="NVT",
        ),
    )


class OsmoticCoefficientConfig(PropertyConfig):
    """
    Configuration for the Osmotic Coefficient property calculation.
    """

    name: typing.ClassVar[typing.Literal["osmotic_coefficient"]] = "osmotic_coefficient"

    force_name: typing.ClassVar[typing.Literal["flat_bottom_restraint"]] = "flat_bottom_restraint"
    force_axis: typing.ClassVar[typing.Literal["x", "y", "z"]] = "z"

    bulk: PhaseConfig = Field(
        default_factory=_generate_osmotic_coefficient_config,
        description="Workflow configuration for bulk osmotic coefficient calculations."
    )
