import pathlib
import typing

import numpy as np
import pydantic
from openff.interchange.components._packmol import UNIT_CUBE
from openff.units import unit

from ...utils.pydantic import OpenFFQuantity
from .base import ProtocolConfig


class CoordinateGenerationConfig(ProtocolConfig):
    """
    Configuration for the CoordinateGeneration protocol.
    """

    name: typing.ClassVar[str] = "CoordinateGeneration"
    n_max_molecules: int = pydantic.Field(
        default=1000, description="Number of molecules to pack"
    )

    coordinate_store_path: pathlib.Path = pydantic.Field(
        default="coordinate_store.sqlite",
        description="Path to query and store the generated coordinates.",
    )

    packing_tolerance: OpenFFQuantity[unit.angstrom] = pydantic.Field(
        default=2.0 * unit.angstrom, description="Tolerance for packing in Angstroms"
    )
    target_density: OpenFFQuantity[unit.grams / unit.milliliter] = pydantic.Field(
        default=0.8 * unit.grams / unit.milliliter,
        description=(
            "Target density for the packed box in g/mL. "
            "We purposely set this quite low to avoid packing or simulation failures. "
            "Tighten to improve equilibration times, but be wary of packing failures."
        )
    )

    box_shape: np.array = pydantic.Field(
        default=np.array(UNIT_CUBE),
        description=(
            "Unit box vectors to use for packing. See openff.interchange.components._packmol.UNIT_CUBE for reference. "
            "If a different box shape is desired, modify the UNIT_CUBE accordingly. "
        )
    )

    store_packed_coordinates: bool = pydantic.Field(
        default=True,
        description=(
            "Whether to store the packed coordinates in the coordinate store. "
            "If False, coordinates will be generated but not stored. "
            "If saved, the force_field_id will be 'packmol'. "
        ),
    )

    query_stored_coordinates: bool = pydantic.Field(
        default=True,
        description=(
            "Whether to query the coordinate store for existing coordinates "
            "before generating new ones."
        ),
    )

    temperature_tolerance: OpenFFQuantity[unit.kelvin] = pydantic.Field(
        default=1.0 * unit.kelvin,
        description="Tolerance for matching temperature in Kelvin when querying stored coordinates.",
    )

    pressure_tolerance: OpenFFQuantity[unit.atm] = pydantic.Field(
        default=0.01 * unit.atm,
        description="Tolerance for matching pressure in atm when querying stored coordinates.",
    )

    @pydantic.field_validator("coordinate_store_path")
    def _validate_coordinate_store_path(
        cls, v: pathlib.Path
    ) -> pathlib.Path:
        """
        Validate the path to the coordinate store. It returns a resolved pathlib.Path object.

        This method ensures that:
        - The path is a valid pathlib.Path object
        - It either does not already exist, or is a compatible SQLite database file
        """

        ...
