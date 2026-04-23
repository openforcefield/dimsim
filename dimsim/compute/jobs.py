import hashlib
import json
import os


def make_job_id(input_file, forcefield, temperature, pressure, seed):
    params = {
        "input_file": input_file,
        "forcefield": forcefield,
        "temperature": float(temperature),
        "pressure": float(pressure),
        "seed": seed,
    }
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(params_str.encode()).hexdigest()[:12]


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
