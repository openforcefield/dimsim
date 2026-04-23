import parsl

from dimsim.compute.apps import analyze_trajectory, create_initial_configuration, run_simulation
from dimsim.compute.jobs import get_job_paths, is_complete, make_job_id


class SimulationWorkflow:
    def __init__(self, base_dir, parsl_config):
        self.base_dir = base_dir
        parsl.load(parsl_config)

    def submit(self, input_file, forcefield, temperature, pressure, seed=42):
        """Submit a single end-to-end simulation pipeline."""
        job_id = make_job_id(input_file, forcefield, temperature, pressure, seed)
        job_dir = get_job_paths(self.base_dir, job_id)["root"]

        if is_complete(self.base_dir, job_id):
            return None  # already done, skip

        config_future = create_initial_configuration(input_file, forcefield, temperature, pressure)
        sim_future = run_simulation(config_future, job_dir)
        analysis_future = analyze_trajectory(sim_future, job_dir)

        return {"job_id": job_id, "future": analysis_future}

    def submit_batch(self, job_specs):
        """Submit many jobs, skipping already-complete ones."""
        return [result for spec in job_specs if (result := self.submit(**spec)) is not None]

    def run(self, job_specs):
        """Submit a batch and block until all complete."""
        pending = self.submit_batch(job_specs)

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
