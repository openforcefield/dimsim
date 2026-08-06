import pathlib
from collections.abc import Sequence

import parsl
import pyarrow.parquet

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
from dimsim.configs.liquid import LIQUID_SCHEMA
from dimsim.configs.targets.thermo import DataEntry


class SimulationWorkflow:
    def __init__(self, base_dir, parsl_config):
        pathlib.Path(base_dir).mkdir(exist_ok=True)

        self.base_dir = base_dir

        parsl.load(parsl_config)

        # this should be based on a schema that's not specific to liquids
        pyarrow.parquet.write_table(LIQUID_SCHEMA.empty_table(), "jobs.parquet")

    def _submit_compute(
        self,
        compute_config: BaseComputeConfig,
    ):
        """Submit a single end-to-end simulation pipeline."""
        import logging

        logging.info("Starting packing app")
        logging.info(f"Submitting {compute_config} compute configs to workflow")

        job_id = make_job_id(compute_config)
        job_dir = get_job_paths(self.base_dir, job_id)["root"]
        # maybe serialize all configs into the job_dir? could simplify some function signatures
        pathlib.Path(job_dir).mkdir(exist_ok=True)

        logging.info(f"Made job id (same as job dir) {job_id} for this compute config")

        if pathlib.Path(job_dir, "production_trajectory.dcd").exists():
            logging.info(f"short-circuiting {job_id}!")
            return None  # already done, skip

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
        import logging

        logging.info(f"Submitting {len(compute_configs)} compute configs to workflow")
        return [result for spec in compute_configs if (result := self._submit_compute(spec)) is not None]

    def submit_target(
        self,
        target_config: DataEntry,
        force_field: str,
        n_molecules: int,
    ):
        import logging

        from dimsim.compute.prep import (
            _compute_configs_from_data_entry,
        )

        logging.info(
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
