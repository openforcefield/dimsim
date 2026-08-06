from __future__ import annotations

import json

from openff.toolkit import Topology
from parsl import File

from dimsim.compute._files import MinimizationFiles, PreparingFiles
from dimsim.configs.liquid import BulkLiquid


def _minimize_energy(
    system_future: dict[str, PreparingFiles],
    job_dir: str,
) -> dict[str, float | MinimizationFiles]:
    import logging
    import pathlib

    import openmm.app
    import openmm.unit
    from openff.toolkit import Molecule

    logger = logging.getLogger("dimsim")  # same package name
    logger.handlers.clear()  # worker starts fresh, but be safe
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"{job_dir}/minimize.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    logger.info("Starting OpenMM energy minimization app")

    files = MinimizationFiles(
        topology=File(f"{job_dir}/minimized_topology.pdb"),
        system=File(f"{job_dir}/minimized_system.xml"),
        integrator=File(f"{job_dir}/minimized_integrator.xml"),
        checkpoint=File(f"{job_dir}/minimized_checkpoint.chk"),
    )

    system = openmm.XmlSerializer.deserialize(open(system_future["prepared_files"]["openmm_system"].filepath).read())

    compute_config: BulkLiquid = BulkLiquid(**json.load(open(f"{job_dir}/compute_config.json")))  # type: ignore[typeddict-item]

    # this should be in Kelvin
    temperature = compute_config["temperature"]

    molecules = [Molecule.from_smiles(smiles) for smiles in compute_config["smiles"]]

    topology = Topology.from_pdb(
        system_future["prepared_files"]["packed_topology"].filepath,
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

    with open(files["topology"].filepath, "w") as f:
        openmm.app.PDBFile.writeFile(simulation.topology, final_state.getPositions(), f)

    original: float = original_state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
    final: float = final_state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

    logger.info(f"Minimized energy from {original:.2f} to {final:.2f}")

    if not pathlib.Path(files["topology"].filepath).exists():
        raise FileNotFoundError(f"Topology file {files['topology'].filepath} does not exist")

    with open(files["topology"].filepath, "w") as f:
        openmm.app.PDBFile.writeFile(
            topology=simulation.topology,
            positions=simulation.context.getState(getPositions=True).getPositions(),
            file=f,
        )

    # Do we need this? the system is probably the same before and after minimization ...
    with open(files["system"].filepath, "w") as f:
        f.write(openmm.XmlSerializer.serialize(simulation.system))

    # same as above
    with open(files["integrator"].filepath, "w") as f:
        f.write(openmm.XmlSerializer.serialize(simulation.integrator))

    simulation.saveCheckpoint(files["checkpoint"].filepath)

    return {
        "simulation_files": files,
        "original": original,
        "final": final,
    }
