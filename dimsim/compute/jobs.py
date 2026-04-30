import hashlib
import json
import os

from dimsim.configs.compute.density import DensityConfig


def make_job_id(compute_config: DensityConfig) -> str:
    """
    Generate a deterministic job ID from compute parameters.

    Maybe this could be simplified if it only has one argument?
    """
    params = {
        "tag": compute_config["tag"],
        "target": compute_config["target"],
        "force_field": compute_config["force_field"],
        "n_molecules": str(compute_config["n_molecules"]),
    }

    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def get_job_paths(base_dir, job_id):
    return {
        "root": f"{base_dir}/{job_id}",
        "checkpoint": f"{base_dir}/{job_id}/checkpoint.chk",
        "trajectory": f"{base_dir}/{job_id}/trajectory.dcd",
        "log": f"{base_dir}/{job_id}/simulation.log",
    }


def is_complete(base_dir, job_id):
    """Check if a job already has complete outputs."""
    return os.path.exists(get_job_paths(base_dir, job_id)["trajectory"])
