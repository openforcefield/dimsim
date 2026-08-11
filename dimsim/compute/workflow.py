import json
import logging
import pathlib
from collections.abc import Sequence

import parsl

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
            job_id, production_future = self._run_liquid_workflow(compute_config)
        elif compute_config["tag"] == "gas":  # type: ignore[typeddict-item]
            job_id, production_future = self._run_gas_workflow(compute_config)
        else:
            raise ValueError(f"Unknown compute config tag {compute_config['tag']}")  # type: ignore[typeddict-item]

        job_dir = get_job_paths(self.base_dir, job_id)["root"]

        # TODO: Switch out into each different property
        analysis_future = run_density_analysis(
            production_future=production_future,
            job_dir=job_dir,
        )

        return {"job_id": job_id, "future": analysis_future}

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

        return self.run(compute_configs=compute_configs)

    def submit_target_batch(
        self,
        target_configs: list[DataEntry],
        force_field: str,
        n_molecules: int,
    ):

        return [
            result
            for spec in target_configs
            if (result := self.submit_target(spec, force_field, n_molecules)) is not None
        ]

    def run(self, compute_configs: Sequence[BaseComputeConfig]):
        """Submit a batch and block until all complete."""
        pending = self._submit_compute_batch(compute_configs)

        results = []
        for item in pending:
            try:
                result = item["future"].result()
                results.append({"job_id": item["job_id"], "result": result})
            except Exception as e:
                results.append({"job_id": item["job_id"], "error": str(e)})

        return results

    def _run_liquid_workflow(self, compute_config: BaseComputeConfig) -> tuple[str, dict[str, ProductionFiles]]:
        job_id = make_job_id(compute_config)
        job_dir = get_job_paths(self.base_dir, job_id)["root"]

        pathlib.Path(job_dir).mkdir(exist_ok=True)

        logger.info(f"Made job id (same as job dir) {job_id} for this compute config")

        json.dump(
            compute_config,
            open(f"{job_dir}/compute_config.json", "w"),
            indent=4,
        )

        if pathlib.Path(job_dir, "production_trajectory.dcd").exists():
            logger.info(f"short-circuiting {job_id}!")
            # already done, skip

            # maybe the short-circuiting should be within apps?
            # otherwise don't know how to construct the ProductionFiles object ...
            return None  # type:ignore[return-value]
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

        return job_id, production_future

    def _run_gas_workflow(self, compute_config: BaseComputeConfig) -> tuple[str, dict[str, ProductionFiles]]:
        assert compute_config["n_molecules"] == 1, (
            f"Gas workflow only supports single-molecule simulations, but got {compute_config['n_molecules']=}"
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

        if pathlib.Path(job_dir, "production_trajectory.dcd").exists():
            logger.info(f"short-circuiting {job_id}!")
            # already done, skip

            # maybe the short-circuiting should be within apps?
            # otherwise don't know how to construct the ProductionFiles object ...
            return None  # type:ignore[return-value]
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

        return job_id, production_future

    def shutdown(self):
        parsl.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
