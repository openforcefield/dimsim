from __future__ import annotations

import openmm
import openmm.app
from openff.toolkit import Topology
from parsl import File

from dimsim.compute._files import MinimizationFiles
from dimsim.configs.liquid import BulkLiquid


def _minimize_energy(
    system_future: dict[str, BulkLiquid | openmm.System], job_dir: str
) -> dict[str, BulkLiquid | float | MinimizationFiles]:
    import logging
    import pathlib

    import openmm
    import openmm.app
    import openmm.unit
    from openff.toolkit import Molecule

    logging.basicConfig(
        filename=f"{job_dir}/simulation.log",
        level=logging.INFO,
    )
    logging.info("Starting OpenMM energy minimization app")
    system = system_future["openmm_system"]

    compute_config: BulkLiquid = system_future["compute_config"]

    # this should be in Kelvin
    temperature = compute_config["temperature"]

    # this file is not in inputs, but is likely to be there by chance -
    # should be a better way of handling this file and the information it stores
    packed_topology_file = f"{job_dir}/packed_topology.pdb"
    minimized_topology_file = f"{job_dir}/minimized_topology.pdb"

    molecules = [Molecule.from_smiles(smiles) for smiles in compute_config["smiles"]]

    topology = Topology.from_pdb(
        packed_topology_file,
        unique_molecules=molecules,
    )

    simulation = openmm.app.Simulation(
        topology=topology.to_openmm(),
        system=system,
        integrator=openmm.LangevinMiddleIntegrator(
            temperature * openmm.unit.kelvin,
            1.0 / openmm.unit.picosecond,
            1.0 * openmm.unit.femtoseconds,  # TODO: This should be user input
        ),
    )

    simulation.context.setPositions(topology.get_positions().to_openmm())

    original_state = simulation.context.getState(energy=True)

    simulation.minimizeEnergy()

    final_state = simulation.context.getState(energy=True, positions=True)

    with open(minimized_topology_file, "w") as f:
        openmm.app.PDBFile.writeFile(simulation.topology, final_state.getPositions(), f)

    original: float = original_state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
    final: float = final_state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

    logging.info(f"Minimized energy from {original:.2f} to {final:.2f}")

    files = MinimizationFiles(
        topology=File(minimized_topology_file),
        system=File(f"{job_dir}/minimized_system.xml"),
        integrator=File(f"{job_dir}/minimized_integrator.xml"),
        checkpoint=File(f"{job_dir}/minimized_checkpoint.chk"),
    )

    if not pathlib.Path(files["topology"].filepath).exists():
        raise FileNotFoundError(f"Topology file {files['topology'].filepath} does not exist")

    with open(files["topology"].filepath, "w") as f:
        openmm.app.PDBFile.writeFile(
            topology=simulation.topology,
            positions=simulation.context.getState(getPositions=True).getPositions(),
            file=f,
        )

    with open(files["system"].filepath, "w") as f:
        f.write(openmm.XmlSerializer.serialize(simulation.system))

    with open(files["integrator"].filepath, "w") as f:
        f.write(openmm.XmlSerializer.serialize(simulation.integrator))

    simulation.saveCheckpoint(files["checkpoint"].filepath)

    return {
        "compute_config": compute_config,
        "simulation_files": files,
        "original": original,
        "final": final,
    }
