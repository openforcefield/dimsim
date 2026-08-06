import json
import logging
import pathlib
from collections.abc import Sequence

import parsl

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
        pathlib.Path(base_dir).mkdir(exist_ok=True)

        self.base_dir = base_dir

        handler = logging.FileHandler(f"{base_dir}/workflow.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # avoid double-logging if root also has handlers

        parsl.load(parsl_config)

    def _submit_compute(
        self,
        compute_config: BaseComputeConfig,
    ):
        """Submit a single end-to-end simulation pipeline."""
        parsl.set_file_logger(f"{self.base_dir}/parsl_log.log", level=logging.DEBUG)

        logger.info("Starting packing app")
        logger.info(f"Submitting {compute_config} compute configs to workflow")

        job_id = make_job_id(compute_config)
        job_dir = get_job_paths(self.base_dir, job_id)["root"]
        # maybe serialize all configs into the job_dir? could simplify some function signatures
        pathlib.Path(job_dir).mkdir(exist_ok=True)

        logger.info(f"Made job id (same as job dir) {job_id} for this compute config")

        json.dump(
            compute_config,
            open(f"{job_dir}/compute_config.json", "w"),
            indent=4,
        )

        if pathlib.Path(job_dir, "production_trajectory.dcd").exists():
            logger.info(f"short-circuiting {job_id}!")
            return None  # already done, skip
        else:
            logger.info(f"short-circuit check for job {job_id} failed, running full workflow")

        # packed_pdb = File(f"{job_dir}/packed.pdb")
        # system_xml = File(f"{job_dir}/system.xml")
        # trajectory_dcd = File(f"{job_dir}/trajectory.dcd")

        # 1. pack from compute config
        pack_future = prepare_packed_topology(compute_config, job_dir)

        # 2. set up openmm system
        setup_future = prepare_openmm_system(pack_future, job_dir)

        # 3. (for now ...) get minimized energy
        minimize_future = minimize_energy(setup_future, job_dir)

        # 4. run equilibration step
        equilibration_future = run_equilibration(
            compute_config=compute_config,
            equilibration_config=None,
            minimization_future=minimize_future,
            job_dir=job_dir,
        )

        # 5. run "production" step
        production_future = run_production(
            compute_config=compute_config,
            production_config=None,
            equilibration_future=equilibration_future,
            job_dir=job_dir,
        )

        # sim_future = run_simulation(config_future, job_dir)
        # 5. analyze trajectory
        # 6. check for convergence, if not converged, run more production and repeat
        # analysis_future = analyze_trajectory(sim_future, job_dir)

        # TODO: Switch out into each different property
        analysis_future = run_density_analysis(
            compute_config=compute_config, production_future=production_future, job_dir=job_dir
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

    def shutdown(self):
        parsl.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
