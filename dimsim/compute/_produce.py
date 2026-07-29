from __future__ import annotations

import openmm
import openmm.app
from parsl import File
from smee.mm import TensorReporter

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
        File(f"{job_dir}/production_trajectory.msgpack"),
    ]

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
