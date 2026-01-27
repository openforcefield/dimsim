"""Property types for molecular simulations.

Each property type defines the simulation requirements and default units
for a specific physical property.

PropertyTypes are singleton classes for now.
"""

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

PropertyName = Literal["density", "hvap", "hmix", "osmotic_coefficient"]
PhaseName = Literal["bulk", "pure", "vacuum"]

if TYPE_CHECKING:
    from ..configs.properties import (
        DensityConfig,
        EnthalpyOfMixingConfig,
        EnthalpyOfVaporizationConfig,
        OsmoticCoefficientConfig,
        ProtocolConfig,
    )

@dataclass(frozen=True)
class PropertyType:
    """Base class for property types defining simulation requirements.

    Attributes
    ----------
    requires_bulk_sim : bool
        Whether this property requires a bulk phase simulation.
    requires_pure_sim : bool
        Whether this property requires a pure component simulation.
    requires_vacuum_sim : bool
        Whether this property requires a vacuum/gas phase simulation.
    default_units : str
        The default units for this property.
    """
    requires_bulk_sim: bool
    requires_pure_sim: bool
    requires_vacuum_sim: bool
    default_units: str

    def generate_default_config(self) -> "ProtocolConfig":
        """Generate a default configuration for this property type."""
        raise NotImplementedError("Subclasses must implement generate_default_config method.")

    def calculate_property(
        self,
        entry,
        **kwargs
    ):
        raise NotImplementedError("Subclasses must implement calculate_property method.")


class _PropertyTypeMeta(type):
    """Metaclass to create singleton property types."""
    _instances: ClassVar[dict] = {}

    def __call__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super(_PropertyTypeMeta, cls).__call__()
        return cls._instances[cls]


class Density(metaclass=_PropertyTypeMeta):
    """Density property type.

    Attributes
    ----------
    requires_bulk_sim : bool = True
    requires_pure_sim : bool = False
    requires_vacuum_sim : bool = False
    default_units : str = "g/mL"
    """
    requires_bulk_sim: bool = True
    requires_pure_sim: bool = False
    requires_vacuum_sim: bool = False
    default_units: str = "g/mL"

    def generate_default_config(self) -> "DensityConfig":
        from ..configs.properties.density import DensityConfig
        return DensityConfig()

    def calculate_property(
        self,
        entry,
        bulk_trajectory_paths: list[pathlib.Path] = None,
    ):
        from ..ops.density import DensityOp

        for bulk_trajectory_path in bulk_trajectory_paths:
            if not bulk_trajectory_path.exists():
                raise FileNotFoundError(
                    f"Bulk trajectory path {bulk_trajectory_path} does not exist."
                )
            values, stds, column_names = DensityOp.apply(
                bulk_trajectory_path,
                entry,
            )



class EnthalpyOfVaporization(metaclass=_PropertyTypeMeta):
    """Enthalpy of vaporization property type.

    Attributes
    ----------
    requires_bulk_sim : bool = True
    requires_pure_sim : bool = False
    requires_vacuum_sim : bool = True
    default_units : str = "kJ/mol"
    """
    requires_bulk_sim: bool = True
    requires_pure_sim: bool = False
    requires_vacuum_sim: bool = True
    default_units: str = "kJ/mol"

    def generate_default_config(self) -> "EnthalpyOfVaporizationConfig":
        from ..configs.properties.dhvap import EnthalpyOfVaporizationConfig
        return EnthalpyOfVaporizationConfig()


class EnthalpyOfMixing(metaclass=_PropertyTypeMeta):
    """Enthalpy of mixing property type.

    Attributes
    ----------
    requires_bulk_sim : bool = True
    requires_pure_sim : bool = True
    requires_vacuum_sim : bool = False
    default_units : str = "kJ/mol"
    """
    requires_bulk_sim: bool = True
    requires_pure_sim: bool = True
    requires_vacuum_sim: bool = False
    default_units: str = "kJ/mol"

    def generate_default_config(self) -> "EnthalpyOfMixingConfig":
        from ..configs.properties.dhmix import EnthalpyOfMixingConfig
        return EnthalpyOfMixingConfig()


class OsmoticCoefficient(metaclass=_PropertyTypeMeta):
    """Osmotic coefficient property type.

    Attributes
    ----------
    requires_bulk_sim : bool = True
    requires_pure_sim : bool = False
    requires_vacuum_sim : bool = False
    default_units : str = "dimensionless"
    """
    requires_bulk_sim: bool = True
    requires_pure_sim: bool = False
    requires_vacuum_sim: bool = False
    default_units: str = "dimensionless"

    FORCE_NAME: ClassVar[str] = "flat_bottom_restraint"

    def generate_default_config(self) -> "OsmoticCoefficientConfig":
        from ..configs.properties.osmotic_coefficient import OsmoticCoefficientConfig
        return OsmoticCoefficientConfig()




# Singleton instances - always return the same object
DENSITY = Density()
ENTHALPY_OF_VAPORIZATION = EnthalpyOfVaporization()
ENTHALPY_OF_MIXING = EnthalpyOfMixing()
OSMOTIC_COEFFICIENT = OsmoticCoefficient()

NAMES_TO_PROPERTY_TYPES = {
    "density": DENSITY,
    "hvap": ENTHALPY_OF_VAPORIZATION,
    "hmix": ENTHALPY_OF_MIXING,
    "osmotic_coefficient": OSMOTIC_COEFFICIENT,
}
