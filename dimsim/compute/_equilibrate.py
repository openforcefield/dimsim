from __future__ import annotations

import json

from parsl import File
from smee.mm import TensorReporter

from dimsim.compute._files import (
    EquilibrationFiles,
    MinimizationFiles,
)
from dimsim.configs.liquid import BulkLiquid

EquilibrationConfig = object


def _run_equilibration(
    equilibration_config: EquilibrationConfig,
    minimization_future: dict[str, float | MinimizationFiles],
    job_dir: str,
) -> dict[str, EquilibrationFiles]:
    # TODO: Expose barostat (+ thermostat?) to user

    import openmm
    import openmm.app
    import openmm.unit

    from dimsim.compute._logging import _set_up_logger

    logger = _set_up_logger(f"{job_dir}/equilibrate.log")

    logger.info("Starting equilibration run")

    compute_config = BulkLiquid(**json.load(open(f"{job_dir}/compute_config.json")))  # type: ignore[typeddict-item]

    minimized_files: MinimizationFiles = minimization_future["simulation_files"]  # type: ignore[assignment]

    with open(minimized_files["topology"].filepath) as f:
        topology = openmm.app.PDBFile(f).getTopology()

    with open(minimized_files["system"].filepath) as f:
        system = openmm.XmlSerializer.deserialize(f.read())

    with open(minimized_files["integrator"].filepath) as f:
        integrator = openmm.XmlSerializer.deserialize(f.read())

    simulation = openmm.app.Simulation(topology, system, integrator)
    simulation.loadCheckpoint(minimized_files["checkpoint"].filepath)
    simulation.system.addForce(
        openmm.MonteCarloBarostat(
            compute_config["pressure"] * openmm.unit.kilopascal,
            compute_config["temperature"] * openmm.unit.kelvin,
        )
    )

    logger.info("Reinitializing context (in equilibration step)")
    simulation.context.reinitialize(preserveState=True)

    files = EquilibrationFiles(
        topology=File(f"{job_dir}/equilibrated_topology.pdb"),
        dcd_trajectory=File(f"{job_dir}/equilibration_trajectory.dcd"),
        msgpack_trajectory=File(f"{job_dir}/equilibration_trajectory.msgpack"),
        log=File(f"{job_dir}/equilibration.log"),
        state_data=File(f"{job_dir}/equilibration.csv"),
        system=File(f"{job_dir}/equilibration_system.xml"),
        integrator=File(f"{job_dir}/equilibration_integrator.xml"),
        checkpoint=File(f"{job_dir}/equilibration_checkpoint.chk"),
    )

    simulation.reporters.append(
        openmm.app.StateDataReporter(
            file=files["state_data"].filepath,
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
        file=files["dcd_trajectory"].filepath,
        reportInterval=1000,
    )

    smee_reporter = TensorReporter(
        output_file=open(files["msgpack_trajectory"].filepath, "wb"),
        report_interval=1000,
        beta=1.0 / openmm.unit.kilocalories_per_mole,
        pressure=compute_config["pressure"] * openmm.unit.kilopascal,
    )

    simulation.reporters.append(dcd_reporter)
    simulation.reporters.append(smee_reporter)

    simulation.context.setVelocitiesToTemperature(
        compute_config["temperature"] * openmm.unit.kelvin,
        compute_config["replicate_index"] + 1,
    )

    logger.info("Running 10,000 steps of MD")

    simulation.step(10_000)

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

    # do we no longer need to wire through the compute config ... ?
    return {"simulation_files": files}
