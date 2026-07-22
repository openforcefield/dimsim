from __future__ import annotations

import openmm
import openmm.app
from openff.toolkit import Topology
from parsl import File, python_app

from dimsim.configs.liquid import BulkLiquid


@python_app
def prepare_packed_topology(
    compute_config: BulkLiquid,
    job_dir: str,
) -> dict[str, BulkLiquid | Topology]:
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
        target_density=Quantity(compute_config.get("density", 0.7) * 0.7, "g/mL"),  # type: ignore[operator]
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
    packing_future: dict[str, BulkLiquid | Topology],
    job_dir: str,
) -> dict[str, BulkLiquid | openmm.System]:
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
    system_future: dict[str, BulkLiquid | openmm.System], job_dir: str
) -> dict[str, BulkLiquid | float | tuple[File, ...]]:
    import logging

    import openmm
    import openmm.app
    import openmm.unit
    from openff.toolkit import Molecule, Topology

    logging.basicConfig(
        filename=f"{job_dir}/simulation.log",
        level=logging.INFO,
    )
    logging.info("Starting OpenMM energy minimization app")
    system = system_future["openmm_system"]

    compute_config: BulkLiquid = system_future["compute_config"]

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

    simulation_files = [
        File(f"{job_dir}/simulation_topology.pdb"),
        File(f"{job_dir}/simulation_system.xml"),
        File(f"{job_dir}/simulation_integrator.xml"),
        File(f"{job_dir}/simulation_checkpoint.chk"),
    ]

    with open(simulation_files[0].filepath, "w") as f:
        openmm.app.PDBFile.writeFile(
            topology=simulation.topology,
            positions=simulation.context.getState(getPositions=True).getPositions(),
            file=f,
        )

    with open(simulation_files[1].filepath, "w") as f:
        f.write(openmm.XmlSerializer.serialize(simulation.system))

    with open(simulation_files[2].filepath, "w") as f:
        f.write(openmm.XmlSerializer.serialize(simulation.integrator))

    simulation.saveCheckpoint(simulation_files[3].filepath)

    return {
        "compute_config": compute_config,
        "simulation_files": tuple(simulation_files),
        "original": original,
        "final": final,
    }


EquilibrationConfig = object


@python_app
def run_equilibration(
    compute_config: BulkLiquid,
    equilibration_config: EquilibrationConfig,
    minimization_future: dict[str, BulkLiquid | float | tuple[File, ...]],
    job_dir: str,
) -> dict[str, tuple[File, ...]]:
    # TODO: Expose barostat (+ thermostat?) to user
    import logging

    import openmm
    import openmm.app
    import openmm.unit
    from smee.mm import TensorReporter

    logging.basicConfig(
        filename=f"{job_dir}/simulation.log",
        level=logging.INFO,
    )
    logging.info("Starting equilibration run")

    minimized_files: tuple[File, ...] = minimization_future["simulation_files"]  # type: ignore[assignment]

    with open(minimized_files[0].filepath) as f:
        topology = openmm.app.PDBFile(f).getTopology()

    with open(minimized_files[1].filepath) as f:
        system = openmm.XmlSerializer.deserialize(f.read())

    with open(minimized_files[2].filepath) as f:
        integrator = openmm.XmlSerializer.deserialize(f.read())

    simulation = openmm.app.Simulation(topology, system, integrator)
    simulation.loadCheckpoint(minimized_files[3].filepath)
    simulation.system.addForce(
        openmm.MonteCarloBarostat(
            compute_config["pressure"] * openmm.unit.kilopascal,
            compute_config["temperature"] * openmm.unit.kelvin,
        )
    )

    logging.info("Reinitializing context (in equilibration step)")
    simulation.context.reinitialize(preserveState=True)

    simulation_files = [
        File(f"{job_dir}/equilibrated_topology.pdb"),
        File(f"{job_dir}/equilibration_trajectory.dcd"),
        File(f"{job_dir}/equilibration_log.log"),
        File(f"{job_dir}/production_system.xml"),  # probably don't need to carry this through ...
        File(f"{job_dir}/equilibration_integrator.xml"),
        File(f"{job_dir}/equilibration_checkpoint.chk"),
        File(f"{job_dir}/equilibration_trajectory.msgpack"),
    ]

    simulation.reporters.append(
        openmm.app.StateDataReporter(
            file=simulation_files[2].filepath,
            reportInterval=1000,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
        )
    )

    dcd_reporter = openmm.app.DCDReporter(
        file=simulation_files[1].filepath,
        reportInterval=1000,
    )

    smee_reporter = TensorReporter(
        output_file=open(simulation_files[6].filepath, "wb"),
        report_interval=1000,
        beta=1.0 / openmm.unit.kilocalories_per_mole,
        pressure=compute_config["pressure"] * openmm.unit.kilopascal,
    )

    simulation.reporters.append(dcd_reporter)
    simulation.reporters.append(smee_reporter)

    logging.info("Running 10,000 steps of MD")

    simulation.step(10_000)

    with open(simulation_files[0].filepath, "w") as f:
        openmm.app.PDBFile.writeFile(
            topology=simulation.topology,
            positions=simulation.context.getState(getPositions=True).getPositions(),
            file=f,
        )
    with open(simulation_files[3].filepath, "w") as f:
        f.write(openmm.XmlSerializer.serialize(simulation.system))

    with open(simulation_files[4].filepath, "w") as f:
        f.write(openmm.XmlSerializer.serialize(simulation.integrator))

    simulation.saveCheckpoint(simulation_files[5].filepath)

    # do we no longer need to wire through the compute config ... ?
    return {"simulation_files": tuple(simulation_files)}


ProductionConfig = object


@python_app
def run_production(
    compute_config: BulkLiquid,
    production_config: ProductionConfig,
    equilibration_future: dict[str, tuple[File, ...]],
    job_dir: str,
) -> dict[str, tuple[File, ...]]:
    import logging

    from smee.mm import TensorReporter

    logging.basicConfig(
        filename=f"{job_dir}/simulation.log",
        level=logging.INFO,
    )
    logging.info("Starting production run")

    equilibrated_files: tuple[File, ...] = equilibration_future["simulation_files"]  # type: ignore[assignment]

    with open(equilibrated_files[0].filepath) as f:
        topology = openmm.app.PDBFile(f).getTopology()

    with open(equilibrated_files[3].filepath) as f:
        system = openmm.XmlSerializer.deserialize(f.read())

    with open(equilibrated_files[4].filepath) as f:
        integrator = openmm.XmlSerializer.deserialize(f.read())

    with open(equilibrated_files[5].filepath, "rb") as f:
        simulation = openmm.app.Simulation(topology, system, integrator)
        simulation.loadCheckpoint(f)

    logging.info("Reinitializing context (in production step)")
    simulation.context.reinitialize(preserveState=True)

    simulation_files = [
        File(f"{job_dir}/production_topology.pdb"),
        File(f"{job_dir}/production_trajectory.dcd"),
        File(f"{job_dir}/production_log.log"),
        File(f"{job_dir}/production_system.xml"),
        File(f"{job_dir}/production_integrator.xml"),
        File(f"{job_dir}/production_checkpoint.chk"),
        File(f"{job_dir}/production_trajectory.msgpack"),
    ]

    simulation.reporters.append(
        openmm.app.StateDataReporter(
            file=simulation_files[2].filepath,
            reportInterval=1000,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
        )
    )

    dcd_reporter = openmm.app.DCDReporter(
        file=simulation_files[1].filepath,
        reportInterval=1000,
    )

    smee_reporter = TensorReporter(
        output_file=open(simulation_files[6].filepath, "wb"),
        report_interval=1000,
        beta=1.0 / openmm.unit.kilocalories_per_mole,
        pressure=compute_config["pressure"] * openmm.unit.kilopascal,
    )

    simulation.reporters.append(dcd_reporter)
    simulation.reporters.append(smee_reporter)

    logging.info("Running 100,000 steps of MD")
    # simulation.step(equilibration_config["steps_per_iteration"])
    simulation.step(100_000)

    with open(simulation_files[0].filepath, "w") as f:
        openmm.app.PDBFile.writeFile(
            topology=simulation.topology,
            positions=simulation.context.getState(getPositions=True).getPositions(),
            file=f,
        )
    with open(simulation_files[3].filepath, "w") as f:
        f.write(openmm.XmlSerializer.serialize(simulation.system))

    with open(simulation_files[4].filepath, "w") as f:
        f.write(openmm.XmlSerializer.serialize(simulation.integrator))

    simulation.saveCheckpoint(simulation_files[5].filepath)

    return {"simulation_files": tuple(simulation_files)}


@python_app
def run_density_analysis(
    compute_config: BulkLiquid,
    production_future: dict[str, tuple[File, ...]],
    job_dir: str,
) -> dict[str, float]:
    import logging

    import pandas

    logging.basicConfig(
        filename=f"{job_dir}/simulation.log",
        level=logging.INFO,
    )
    logging.info("Starting density analysis")

    # TODO: Double check for equilibration/stability
    production_files: tuple[File, ...] = production_future["simulation_files"]  # type: ignore[assignment]

    dataframe = pandas.read_csv(production_files[2].filepath)

    estimate = dataframe["Density (g/mL)"].mean()
    logging.info(f"Estimated mean density: {estimate:.4f} g/mL")

    return {"mean_density": estimate}


@python_app
def run_simulation(config, job_dir, steps=10000):
    ...
    return {"trajectory": f"{job_dir}/trajectory.dcd", "energy": -12345.6}


@python_app
def analyze_trajectory(simulation_result, job_dir):
    ...
    return {"mean_energy": ..., "rmsd": ...}
