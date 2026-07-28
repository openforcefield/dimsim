from __future__ import annotations

import openmm
import openmm.app
from parsl import File

from dimsim.compute._files import (
    EquilibrationFiles,
    ProductionFiles,
)
from dimsim.configs.liquid import BulkLiquid

ProductionConfig = object


def _run_production(
    compute_config: BulkLiquid,
    production_config: ProductionConfig,
    equilibration_future: dict[str, EquilibrationFiles],
    job_dir: str,
) -> dict[str, ProductionFiles]:

    import logging

    logging.basicConfig(
        filename=f"{job_dir}/simulation.log",
        level=logging.INFO,
    )
    logging.info("Starting production run")

    equilibrated_files: EquilibrationFiles = equilibration_future["simulation_files"]

    with open(equilibrated_files["topology"].filepath) as f:
        topology = openmm.app.PDBFile(f).getTopology()

    with open(equilibrated_files["system"].filepath) as f:
        system = openmm.XmlSerializer.deserialize(f.read())

    with open(equilibrated_files["integrator"].filepath) as f:
        integrator = openmm.XmlSerializer.deserialize(f.read())

    with open(equilibrated_files["checkpoint"].filepath, "rb") as f:
        simulation = openmm.app.Simulation(topology, system, integrator)
        simulation.loadCheckpoint(f)

    logging.info("Reinitializing context (in production step)")
    simulation.context.reinitialize(preserveState=True)

    files = ProductionFiles(
        topology=File(f"{job_dir}/production_topology.pdb"),
        trajectory=File(f"{job_dir}/production_trajectory.dcd"),
        log=File(f"{job_dir}/production_log.log"),
        system=File(f"{job_dir}/production_system.xml"),
        integrator=File(f"{job_dir}/production_integrator.xml"),
        checkpoint=File(f"{job_dir}/production_checkpoint.chk"),
    )
    simulation.reporters.append(
        openmm.app.StateDataReporter(
            file=files["log"].filepath,
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

    simulation.reporters.append(openmm.app.DCDReporter(files["trajectory"].filepath, 1000))

    logging.info("Running 100,000 steps of MD")
    # simulation.step(equilibration_config["steps_per_iteration"])
    simulation.step(100_000)

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

    return {"simulation_files": files}
