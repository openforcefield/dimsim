from __future__ import annotations

from parsl import File

from dimsim.configs.liquid import BulkLiquid

EquilibrationConfig = object


def _run_equilibration(
    compute_config: BulkLiquid,
    equilibration_config: EquilibrationConfig,
    minimization_future: dict[str, BulkLiquid | float | tuple[File, ...]],
    job_dir: str,
) -> dict[str, tuple[File, ...]]:
    # TODO: Expose barostat (+ thermostat?) to user
    import logging

    import openmm.app
    import openmm.unit

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

    simulation.reporters.append(openmm.app.DCDReporter(simulation_files[1].filepath, 1000))

    logging.info("Running 10,000 steps of MD")
    # simulation.step(equilibration_config["steps_per_iteration"])
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
