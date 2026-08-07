import pathlib

from dimsim.compute.jobs import make_job_id
from dimsim.compute.prep import (
    _compute_configs_from_data_entry,
)
from dimsim.configs.targets.thermo import DataEntry


def fetch_trajectory_paths_from_target(
    base_dir: str,
    target: DataEntry,
    force_field: str,
    n_molecules: int,
    n_replicates: int,
) -> tuple[str, ...]:
    compute_configs = _compute_configs_from_data_entry(
        target,
        force_field,
        n_molecules,
        n_replicates,
    )
    job_ids = [make_job_id(compute_config) for compute_config in compute_configs]

    # TODO: These might not always be named "production_trajectory.dcd"
    trajectories = [pathlib.Path(base_dir) / job_id / "production_trajectory.dcd" for job_id in job_ids]

    for trajectory in trajectories:
        if not trajectory.exists():
            raise FileNotFoundError(f"Trajectory {trajectory} does not exist")

    return tuple([trajectory.as_posix() for trajectory in trajectories])
