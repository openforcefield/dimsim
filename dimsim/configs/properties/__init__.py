"""
Default property configurations with default options designed for ~1000 molecules.
"""
import typing

__all__ = [
    "DensityConfig",
    "EnthalpyOfVaporizationConfig",
    "EnthalpyOfMixingConfig",
    "OsmoticCoefficientConfig",
]

from .base import PropertyConfig
from .density import DensityConfig
from .dhmix import EnthalpyOfMixingConfig
from .dhvap import EnthalpyOfVaporizationConfig
from .osmotic_coefficient import OsmoticCoefficientConfig

PropertyConfigType = typing.Union[
    DensityConfig,
    EnthalpyOfVaporizationConfig,
    EnthalpyOfMixingConfig,
    OsmoticCoefficientConfig,
    PropertyConfig  # TODO: does this allow extensions?
]
