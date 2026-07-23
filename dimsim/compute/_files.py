from typing import TypedDict

from parsl import File


class PackingFiles(TypedDict):
    packed_topology: File


class MinimizationFiles(TypedDict):
    """Files needed in the `minimize_energy` app."""

    topology: File
    system: File
    integrator: File
    checkpoint: File


class EquilibrationFiles(TypedDict):
    """Files needed in the `run_equilibration` app."""

    topology: File
    trajectory: File
    log: File
    system: File
    integrator: File
    checkpoint: File


class ProductionFiles(TypedDict):
    """Files needed in the `run_production` app."""

    topology: File
    trajectory: File
    log: File
    system: File
    integrator: File
    checkpoint: File
