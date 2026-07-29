"""Get observables from production runs."""

import pathlib

import smee
from descent.targets.thermo import SimulationKey, _compute_observables, _Observables
from openff.toolkit import ForceField, Molecule, Topology

from dimsim.configs._compute import BaseComputeConfig


def get_observables(
    compute_config: BaseComputeConfig,
    job_dir: str,
) -> _Observables:
    """Get observables from production runs."""

    # easier if we just serialize the Interchange in each directory, but it sure
    # feels like there should be a better way of handling all of these objects
    topology = Topology.from_pdb(
        f"{job_dir}/production_topology.pdb",
        unique_molecules=[Molecule.from_smiles(smiles) for smiles in compute_config["smiles"]],
    )
    interchanges = [ForceField(compute_config["force_field"]).create_interchange(topology)]

    tensor_ff, tensor_topologies = smee.converters.convert_interchange(interchanges)
    tensor_system = smee.TensorSystem(topologies=tensor_topologies, n_copies=[1], is_periodic=True)

    return _compute_observables(
        phase="bulk" if compute_config["tag"] == "liquid" else "vacuum",
        key=SimulationKey(
            smiles=tuple(compute_config["smiles"]),
            counts=tuple([int(compute_config["n_molecules"] * x) for x in compute_config["x"]]),
            temperature=compute_config["temperature"],
            pressure=compute_config["pressure"],
        ),
        system=tensor_system,
        force_field=tensor_ff,
        output_dir=pathlib.Path(job_dir),
        cached_dir=None,
    )


"""
def _compute_observables(
    phase: Phase,
    key: SimulationKey,
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    output_dir: pathlib.Path,
    cached_dir: pathlib.Path | None,
) -> _Observables:
"""
