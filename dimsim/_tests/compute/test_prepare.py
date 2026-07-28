import json
from importlib.resources import files

import openmm
import pytest
from openff.toolkit import Molecule, Topology

from dimsim.compute._prepare import _prepare_openmm_system
from dimsim.configs.liquid import BulkLiquid


@pytest.fixture
def bulk_liquid():
    return BulkLiquid(
        tag="liquid",
        # TODO: might make tests quicker to have a slimmed-down Sage with fewer parameters
        # (but still use AshGC charges)
        force_field="openff-2.3.0.offxml",
        n_molecules=100,
        smiles=["CC", "O"],
        x=[0.6, 0.4],
        temperature=301.15,
        pressure=101.325,
        density=0.65,  # very low, just for quicker packing
    )


@pytest.fixture
def packing_future() -> dict[str, BulkLiquid | Topology]:
    compute_config = BulkLiquid(
        json.load(
            open(
                files("dimsim") / "_tests/data/app_files/sample_density/compute_config.json",
            ),
        ),
    )

    packed_topology = Topology.from_pdb(
        str(files("dimsim") / "_tests/data/app_files/sample_density/packed_topology.pdb"),
        unique_molecules=[Molecule.from_smiles(smiles) for smiles in compute_config["smiles"]],
    )

    return {
        "compute_config": compute_config,  # do we really need to return this?
        "packed_topology": packed_topology,
    }


def test_prepare_openmm_system(packing_future, tmp_path):
    prepare_result = _prepare_openmm_system(
        packing_future=packing_future,
        job_dir=str(tmp_path),
    )

    assert isinstance(prepare_result["openmm_system"], openmm.System)

    assert prepare_result["openmm_system"].getNumParticles() == packing_future["packed_topology"].n_atoms
