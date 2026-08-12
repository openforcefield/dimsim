import json
import logging
import pathlib
from collections.abc import Sequence

import parsl
from parsl import File

from dimsim.compute._files import (
    ProductionFiles,
)
from dimsim.compute.apps import (
    minimize_energy,
    prepare_openmm_system,
    prepare_packed_topology,
    run_density_analysis,
    run_equilibration,
    run_production,
)
from dimsim.compute.jobs import get_job_paths, make_job_id
from dimsim.configs._compute import BaseComputeConfig
from dimsim.configs.targets.thermo import DataEntry

logger = logging.getLogger(__name__)  # module-level logger, not root


class SimulationWorkflow:
    def __init__(self, base_dir, parsl_config):
        from dimsim.compute._logging import _set_up_logger

        pathlib.Path(base_dir).mkdir(exist_ok=True)

        self.base_dir = base_dir

        # at scale these should become databases or something more robust
        self._targets = dict()
        self._target_compute_mapping: dict[tuple[int, str, int, int], list[str]] = dict()

        # self.logger, maybe?
        logger = _set_up_logger(f"{base_dir}/workflow.log")

        logger.info(f"Initialized SimulationWorkflow with base_dir={base_dir}")

        parsl.load(parsl_config)

        logger.info("Parsl config loaded")

    def _submit_compute(
        self,
        compute_config: BaseComputeConfig,
    ):
        """Submit a single end-to-end simulation pipeline."""
        parsl.set_file_logger(f"{self.base_dir}/parsl_log.log", level=logging.DEBUG)

        logger.info(f"Submitting {compute_config} compute configs to workflow")

        if compute_config["tag"] == "liquid":  # type: ignore[typeddict-item]
            logger.info(f"Submitting {compute_config} compute configs to liquid workflow")
            return self._run_liquid_workflow(compute_config)
        elif compute_config["tag"] == "gas":  # type: ignore[typeddict-item]
            logger.info(f"Submitting {compute_config} compute configs to gas workflow")
            return self._run_gas_workflow(compute_config)
        else:
            raise ValueError(f"Unknown compute config tag {compute_config['tag']}")  # type: ignore[typeddict-item]

    def _submit_compute_batch(
        self,
        compute_configs: Sequence[BaseComputeConfig],
    ):
        """Submit many jobs, skipping already-complete ones."""

        logger.info(f"Submitting {len(compute_configs)} compute configs to workflow")
        return [result for spec in compute_configs if (result := self._submit_compute(spec)) is not None]

    def submit_target(
        self,
        target_config: DataEntry,
        force_field: str,
        n_molecules: int,
        n_replicates: int = 3,
    ):

        from dimsim.compute.prep import (
            _compute_configs_from_data_entry,
        )

        self._targets[target_config["id"]] = target_config

        logger.info(
            f"submitting target {target_config['tag']} with {n_molecules} molecules and force field {force_field}"
        )
        # for some properties this will be len 2+, for some len 1,
        # but just treat it as an iterable either way
        compute_configs = _compute_configs_from_data_entry(
            target_config,
            force_field,
            n_molecules,
            n_replicates=n_replicates,
        )

        # track targets (+ other arg) mapped to compute runs (as job_id) so we can look up results later
        self._target_compute_mapping[(target_config["id"], force_field, n_molecules, n_replicates)] = [
            make_job_id(compute_config) for compute_config in compute_configs
        ]

        # run each compute job - can be >1 compute job per property
        self.run(compute_configs=compute_configs)

    def submit_target_batch(
        self,
        target_configs: list[DataEntry],
        force_field: str,
        n_molecules: int,
        n_replicates: int = 3,
    ):

        return [
            result
            for spec in target_configs
            if (result := self.submit_target(spec, force_field, n_molecules, n_replicates)) is not None
        ]

    def estimate_target(
        self,
        target_config: DataEntry,
        force_field: str,
        n_molecules: int,
        n_replicates: int = 3,
    ):
        """Naively estimate a target property, assuming the compute has already been run."""
        if target_config["tag"] == "density":
            job_id = self._target_compute_mapping[(target_config["id"], force_field, n_molecules, n_replicates)][0]
            return run_density_analysis(
                job_dir=str(pathlib.Path(self.base_dir) / job_id),
            )

    def run(self, compute_configs: Sequence[BaseComputeConfig]):
        """Submit a batch and block until all complete."""
        pending = self._submit_compute_batch(compute_configs)

        results = []
        for item in pending:
            try:
                result = item["future"].result()
                results.append({"job_id": item["job_id"], "result": result})
            except Exception as e:
                print(f"{item.keys()=}")
                print(f"error is {e=}")
                results.append({"job_id": item["job_id"], "error": str(e)})

        return results

    def _run_liquid_workflow(self, compute_config: BaseComputeConfig) -> dict[str, str | dict[str, ProductionFiles]]:
        job_id = make_job_id(compute_config)
        job_dir = get_job_paths(self.base_dir, job_id)["root"]

        pathlib.Path(job_dir).mkdir(exist_ok=True)

        logger.info(f"Made job id (same as job dir) {job_id} for this compute config")

        json.dump(
            compute_config,
            open(f"{job_dir}/compute_config.json", "w"),
            indent=4,
        )

        if False and pathlib.Path(job_dir, "production_trajectory.dcd").exists():
            logger.info(f"short-circuiting {job_id}!")
            # already done, skip

            # bit of a hack + assumes some file structure
            files = ProductionFiles(
                topology=File(f"{job_dir}/production_topology.pdb"),
                dcd_trajectory=File(f"{job_dir}/production_trajectory.dcd"),
                msgpack_trajectory=File(f"{job_dir}/production_trajectory.msgpack"),
                log=File(f"{job_dir}/production.log"),
                state_data=File(f"{job_dir}/production.csv"),
                system=File(f"{job_dir}/production_system.xml"),
                integrator=File(f"{job_dir}/production_integrator.xml"),
                checkpoint=File(f"{job_dir}/production_checkpoint.chk"),
            )

            return {"job_id": job_id, "future": {"files": files}}
        else:
            logger.info(f"short-circuit check for job {job_id} failed, running full workflow")

        pack_future = prepare_packed_topology(job_dir)

        setup_future = prepare_openmm_system(pack_future, job_dir)

        minimize_future = minimize_energy(setup_future, job_dir)

        equilibration_future = run_equilibration(
            equilibration_config=None,
            minimization_future=minimize_future,
            job_dir=job_dir,
        )

        production_future = run_production(
            production_config=None,
            equilibration_future=equilibration_future,
            job_dir=job_dir,
        )

        return {"job_id": job_id, "future": production_future}

    def _run_gas_workflow(self, compute_config: BaseComputeConfig) -> dict[str, str | dict[str, ProductionFiles]]:
        assert compute_config["n_molecules"] == 1, (
            f"Gas workflow only supports single-molecule simulations, but got {compute_config['n_molecules']=}"
            f"and, more generally, {compute_config=}"
        )

        job_id = make_job_id(compute_config)
        job_dir = get_job_paths(self.base_dir, job_id)["root"]

        pathlib.Path(job_dir).mkdir(exist_ok=True)

        logger.info(f"Made job id (same as job dir) {job_id} for this compute config")

        json.dump(
            compute_config,
            open(f"{job_dir}/compute_config.json", "w"),
            indent=4,
        )

        if False and pathlib.Path(job_dir, "production_trajectory.dcd").exists():
            logger.info(f"short-circuiting {job_id}!")
            # already done, skip

            # bit of a hack + assumes some file structure
            files = ProductionFiles(
                topology=File(f"{job_dir}/production_topology.pdb"),
                dcd_trajectory=File(f"{job_dir}/production_trajectory.dcd"),
                msgpack_trajectory=File(f"{job_dir}/production_trajectory.msgpack"),
                log=File(f"{job_dir}/production.log"),
                state_data=File(f"{job_dir}/production.csv"),
                system=File(f"{job_dir}/production_system.xml"),
                integrator=File(f"{job_dir}/production_integrator.xml"),
                checkpoint=File(f"{job_dir}/production_checkpoint.chk"),
            )

            return {"job_id": job_id, "future": {"files": files}}
        else:
            logger.info(f"short-circuit check for job {job_id} failed, running full workflow")

        pack_future = prepare_packed_topology(job_dir)

        setup_future = prepare_openmm_system(pack_future, job_dir)

        minimize_future = minimize_energy(setup_future, job_dir)

        equilibration_future = run_equilibration(
            equilibration_config=None,
            minimization_future=minimize_future,
            job_dir=job_dir,
        )

        production_future = run_production(
            production_config=None,
            equilibration_future=equilibration_future,
            job_dir=job_dir,
        )

        return {"job_id": job_id, "future": production_future}

    def shutdown(self):
        parsl.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
