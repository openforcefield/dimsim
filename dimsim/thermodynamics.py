"""
Defines an API for defining thermodynamic states.
"""

from enum import Enum


class Ensemble(Enum):
    """An enum describing the supported thermodynamic ensembles."""

    NVT = "NVT"
    NPT = "NPT"
