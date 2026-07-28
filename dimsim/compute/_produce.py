from __future__ import annotations

import openmm
import openmm.app
from parsl import File

from dimsim.configs.liquid import BulkLiquid

ProductionConfig = object


def _run_production(
    compute_config: BulkLiquid,
    production_config: ProductionConfig,
    equilibration_future: dict[str, tuple[File, ...]],
    job_dir: str,
) -> dict[str, tuple[File, ...]]:

    import logging

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
