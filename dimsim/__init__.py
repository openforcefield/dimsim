"""
dimsim
Distributed simulation package
"""

from importlib.metadata import version

from dimsim._constants import CUTOFF_ATTRIBUTE, SWITCH_ATTRIBUTE, EnergyFn, PotentialType
from dimsim._models import (
    NonbondedParameterMap,
    ParameterMap,
    TensorConstraints,
    TensorForceField,
    TensorPotential,
    TensorSystem,
    TensorTopology,
    TensorVSites,
    ValenceParameterMap,
    VSiteMap,
)
from dimsim.geometry import add_v_site_coords, compute_v_site_coords
from dimsim.potentials import compute_energy, compute_energy_potential

__version__ = version("dimsim")

__author__ = "Lily Wang"


__all__ = [
    "CUTOFF_ATTRIBUTE",
    "SWITCH_ATTRIBUTE",
    "EnergyFn",
    "NonbondedParameterMap",
    "ParameterMap",
    "PotentialType",
    "TensorConstraints",
    "TensorForceField",
    "TensorPotential",
    "TensorSystem",
    "TensorTopology",
    "TensorVSites",
    "VSiteMap",
    "ValenceParameterMap",
    "__version__",
    "add_v_site_coords",
    "compute_energy",
    "compute_energy_potential",
    "compute_v_site_coords",
]
