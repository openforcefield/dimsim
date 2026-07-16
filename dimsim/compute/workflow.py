import pathlib

import parsl

from dimsim.compute.apps import (
    minimize_energy,
    prepare_openmm_system,
    prepare_packed_topology,
)
from dimsim.compute.jobs import get_job_paths, is_complete, make_job_id
from dimsim.configs._compute import BaseComputeConfig


class SimulationWorkflow:
    def __init__(self, base_dir, parsl_config):
        pathlib.Path(base_dir).mkdir(exist_ok=True)

        self.base_dir = base_dir

        parsl.load(parsl_config)

    def submit(self, compute_config: BaseComputeConfig):
        """Submit a single end-to-end simulation pipeline."""
        job_id = make_job_id(compute_config)
        job_dir = get_job_paths(self.base_dir, job_id)["root"]

        if is_complete(self.base_dir, job_id):
            return None  # already done, skip

        # packed_pdb = File(f"{job_dir}/packed.pdb")
        # system_xml = File(f"{job_dir}/system.xml")
        # trajectory_dcd = File(f"{job_dir}/trajectory.dcd")

        # 1. pack from compute config
        pack_future = prepare_packed_topology(compute_config, job_dir)

        # 2. set up openmm system
        setup_future = prepare_openmm_system(pack_future, job_dir)

        # 3 (for now ...) get minimized energy
        minimize_future = minimize_energy(setup_future, job_dir)

        # 3. run equilibration step
        # 4. run "production" step
        # sim_future = run_simulation(config_future, job_dir)
        # 5. analyze trajectory
        # 6. check for convergence, if not converged, run more production and repeat
        # analysis_future = analyze_trajectory(sim_future, job_dir)

        return {"job_id": job_id, "future": minimize_future}

    def submit_batch(self, compute_configs: list[BaseComputeConfig]):
        """Submit many jobs, skipping already-complete ones."""
        return [result for spec in compute_configs if (result := self.submit(spec)) is not None]

    def run(self, compute_configs: list[BaseComputeConfig]):
        """Submit a batch and block until all complete."""
        pending = self.submit_batch(compute_configs)

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
