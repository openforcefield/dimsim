from __future__ import annotations

import openmm
import openmm.app
from openff.toolkit import Topology
from parsl import File, python_app

from dimsim.compute._equilibrate import EquilibrationConfig
from dimsim.compute._produce import ProductionConfig
from dimsim.configs.liquid import BulkLiquid


@python_app
def prepare_packed_topology(
    compute_config: BulkLiquid,
    job_dir: str,
) -> dict[str, BulkLiquid | Topology]:
    from dimsim.compute._pack import _prepare_packed_topology

    return _prepare_packed_topology(compute_config, job_dir)


@python_app
def prepare_openmm_system(
    packing_future: dict[str, BulkLiquid | Topology],
    job_dir: str,
) -> dict[str, BulkLiquid | openmm.System]:
    from dimsim.compute._prepare import _prepare_openmm_system

    return _prepare_openmm_system(packing_future, job_dir)


@python_app
def minimize_energy(
    system_future: dict[str, BulkLiquid | openmm.System], job_dir: str
) -> dict[str, BulkLiquid | float | tuple[File, ...]]:
    from dimsim.compute._minimize import _minimize_energy

    return _minimize_energy(system_future, job_dir)


@python_app
def run_equilibration(
    compute_config: BulkLiquid,
    equilibration_config: EquilibrationConfig,
    minimization_future: dict[str, BulkLiquid | float | tuple[File, ...]],
    job_dir: str,
) -> dict[str, tuple[File, ...]]:
    from dimsim.compute._equilibrate import _run_equilibration

    return _run_equilibration(compute_config, equilibration_config, minimization_future, job_dir)


@python_app
def run_production(
    compute_config: BulkLiquid,
    production_config: ProductionConfig,
    equilibration_future: dict[str, tuple[File, ...]],
    job_dir: str,
) -> dict[str, tuple[File, ...]]:

    from dimsim.compute._produce import _run_production

    return _run_production(compute_config, production_config, equilibration_future, job_dir)


@python_app
def run_density_analysis(
    compute_config: BulkLiquid,
    production_future: dict[str, tuple[File, ...]],
    job_dir: str,
) -> dict[str, float]:
    """Run a naive density analysis of production trajectories. For debugging only, not for tensor fitting."""
    from dimsim.compute._analyze import _run_density_analysis

    return _run_density_analysis(compute_config, production_future, job_dir)
