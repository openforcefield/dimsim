from __future__ import annotations

from openff.toolkit import Topology

from dimsim.configs.liquid import BulkLiquid


def _prepare_packed_topology(
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
    packed_topology_file = f"{job_dir}/packed_topology.pdb"

    molecules = [Molecule.from_smiles(smiles) for smiles in compute_config["smiles"]]

    if pathlib.Path(packed_topology_file).exists():
        logging.info(f"File {packed_topology_file} already exists, skipping packing.")
        return {
            "compute_config": compute_config,
            "packed_topology": Topology.from_pdb(packed_topology_file, unique_molecules=molecules),
        }

    n_copies = [int(n_molecules * x) for x in compute_config["x"]]

    density = compute_config.get("density")
    if density is None:
        density = 1.0

    result = pack_box(
        molecules,
        n_copies,
        target_density=Quantity(density * 0.7, "g/mL"),
        working_directory=job_dir,
    )

    result.to_file(packed_topology_file, file_format="pdb")

    logging.info(f"packed {result.n_molecules} molecules")

    return {
        "compute_config": compute_config,  # do we really need to return this?
        "packed_topology": result,
    }
