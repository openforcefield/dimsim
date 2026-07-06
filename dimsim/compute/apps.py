import openmm
from openff.toolkit import Topology
from parsl.app.app import python_app

from dimsim.configs.compute.density import LiquidConfig


@python_app
def prepare_packed_topology(
    compute_config: LiquidConfig,
    job_dir: str,
) -> dict[str, LiquidConfig | Topology]:
    import logging
    import pathlib
    import time

    from openff.packmol import pack_box
    from openff.toolkit import Molecule, Quantity

    # making the job dir should maybe happen inside of SimulationWorkflow.submit() instead of here?
    pathlib.Path(job_dir).mkdir(exist_ok=True)

    logging.basicConfig(
        filename=f"{job_dir}/simulation.log",
        level=logging.INFO,
    )
    logging.info("Starting packing app")

    n_molecules = compute_config["n_molecules"]

    time.sleep(n_molecules * 0.001)
    filename = f"{job_dir}/packed_topology.pdb"

    molecules = [Molecule.from_smiles(smiles) for smiles in compute_config["smiles"]]

    if pathlib.Path(filename).exists():
        logging.info(f"File {filename} already exists, skipping packing.")
        return {
            "compute_config": compute_config,
            "packed_topology": Topology.from_pdb(filename, unique_molecules=molecules),
        }

    n_copies = [int(n_molecules * x) for x in compute_config["x"]]

    result = pack_box(
        molecules,
        n_copies,
        target_density=Quantity(compute_config["value"] * 0.7, "g/mL"),
        working_directory=job_dir,
    )

    result.to_file(f"{job_dir}/packed_topology.pdb", file_format="pdb")

    logging.info(f"packed {result.n_molecules} molecules")

    return {
        "compute_config": compute_config,
        "packed_topology": result,
    }


@python_app
def prepare_openmm_system(
    packing_future: dict[str, LiquidConfig | Topology],
    job_dir: str,
) -> dict[str, LiquidConfig | openmm.System]:
    import logging
    import pathlib

    import openmm
    from openff.toolkit import ForceField

    logging.basicConfig(
        filename=f"{job_dir}/simulation.log",
        level=logging.INFO,
    )
    logging.info("Starting OpenMM system creation app")

    filename = f"{job_dir}/openmm_system.xml"

    if pathlib.Path(filename).exists():
        logging.warning(f"File {filename} already exists, skipping packing.")
        return {
            "compute_config": packing_future["compute_config"],
            "openmm_system": openmm.XmlSerializer.deserialize(open(filename).read()),
        }

    compute_config = packing_future["compute_config"]
    packed_topology: Topology = packing_future["packed_topology"]

    force_field = ForceField(compute_config["force_field"])

    openmm_system = force_field.create_openmm_system(packed_topology)

    with open(filename, "w") as f:
        f.write(openmm.XmlSerializer.serialize(openmm_system))

    logging.info("made openmm system!")
    return {
        "compute_config": compute_config,
        "openmm_system": openmm_system,
    }


@python_app
def minimize_energy(
    system_future: dict[str, LiquidConfig | openmm.System], job_dir: str
) -> dict[str, LiquidConfig | float]:
    import logging

    import openmm
    import openmm.app
    import openmm.unit
    from openff.toolkit import Molecule, Topology

    logging.basicConfig(
        filename=f"{job_dir}/simulation.log",
        level=logging.INFO,
    )
    logging.info("Starting energy minimization app")
    system = system_future["openmm_system"]

    compute_config = system_future["compute_config"]

    # this should be in Kelvin
    temperature = compute_config["temperature"]

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
            1.0 * openmm.unit.femtoseconds,
        ),
    )

    simulation.context.setPositions(topology.get_positions().to_openmm())

    original_state = simulation.context.getState(energy=True)

    simulation.minimizeEnergy()

    # simulation.reporters.append(openmm.app.DCDReporter(outputs[0].filepath, 100))

    # print("Minimized energy, now running 10,000 steps of dynamics...")
    # simulation.step(10_000)

    final_state = simulation.context.getState(energy=True, positions=True)

    with open(minimized_topology_file, "w") as f:
        openmm.app.PDBFile.writeFile(simulation.topology, final_state.getPositions(), f)

    original: float = original_state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
    final: float = final_state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

    logging.info(f"Minimized energy from {original:.2f} to {final:.2f}")

    return {
        "compute_config": compute_config,
        "original": original,
        "final": final,
    }


@python_app
def run_simulation(config, job_dir, steps=10000):
    ...
    return {"trajectory": f"{job_dir}/trajectory.dcd", "energy": -12345.6}


@python_app
def analyze_trajectory(simulation_result, job_dir):
    ...
    return {"mean_energy": ..., "rmsd": ...}
