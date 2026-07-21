import hashlib
import json
import os

from dimsim.configs._compute import BaseComputeConfig


# TODO: LiquidConfig should have a base class?
#       one for i.e. GasConfig to derive from
def make_job_id(compute_config: BaseComputeConfig) -> str:
    """
    Generate a deterministic job ID from compute parameters.

    Maybe this could be simplified if it only has one argument?
    """
    # TODO: Improve how this handles VacuumGas
    params = {
        "tag": compute_config["tag"],  # type: ignore[typeddict-item]
        "force_field": compute_config["force_field"],
        "n_molecules": str(compute_config["n_molecules"]),
        "smiles": compute_config["smiles"],
        "x": compute_config["x"],
        "temperature": compute_config["temperature"],
        "pressure": compute_config.get("pressure"),
        "density": compute_config.get("density"),
    }

    # some of attributes - maybe the target value? - could possibly be
    # excluded from hashing criteria here
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
