from __future__ import annotations

import openmm
import openmm.app
from openff.toolkit import Topology

from dimsim.configs.liquid import BulkLiquid


def _prepare_openmm_system(
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
