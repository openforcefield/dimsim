import json
import pathlib

import openmm.unit
import smee.converters
import smee.mm
import torch
from openff.interchange import Interchange


def get_ensemble_averages(job_dir: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    compute_config = json.load(open(pathlib.Path(job_dir) / "compute_config.json"))

    temperature = compute_config["temperature"]
    pressure = compute_config["pressure"]

    # smee's converters start from Interchange(s), maybe we should serialize one out in each job directory?
    interchanges = [Interchange.model_validate_json(open(pathlib.Path(job_dir) / "interchange.json")).read()]
    tensor_force_field, topologies = smee.converters.interchange_to_tensor_system(
        interchanges,
    )

    tensor_system = smee.TensorSystem(
        topologies=topologies,
        n_copies=1,
        is_periodic=interchanges[0].box_vectors is not None,
    )

    return smee.mm.compute_ensemble_averages(
        system=tensor_system,  # smee.TensorSystem
        force_field=tensor_force_field,
        tensor_trajectory_path=pathlib.Path(job_dir) / "production_trajectory.msgpack",
        temperature=temperature * openmm.unit.kelvin,
        pressure=pressure * openmm.unit.atmosphere,
    )
