import json
from importlib.resources import files

import openmm
import pytest
from openff.toolkit import Molecule, Topology
from parsl import File

from dimsim.compute._pack import PackingFiles
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

    return {
        "compute_config": compute_config,  # do we really need to return this?
        "packed_files": PackingFiles(
            packed_topology=File(files("dimsim") / "_tests/data/app_files/sample_density/packed_topology.pdb")
        ),
    }


def test_prepare_openmm_system(packing_future, tmp_path):
    json.dump(
        packing_future["compute_config"],
        open(f"{tmp_path}/compute_config.json", "w"),
    )

    prepare_result = _prepare_openmm_system(
        packing_future=packing_future,
        job_dir=str(tmp_path),
    )

    assert isinstance(prepare_result["prepared_files"], dict)
    assert isinstance(prepare_result["prepared_files"]["openmm_system"], File)

    topology = Topology.from_pdb(
        file_path=packing_future["packed_files"]["packed_topology"].filepath,
        unique_molecules=[Molecule.from_smiles(smiles) for smiles in packing_future["compute_config"]["smiles"]],
    )

    openmm_system = openmm.XmlSerializer.deserialize(
        open(prepare_result["prepared_files"]["openmm_system"].filepath).read()
    )

    assert openmm_system.getNumParticles() == topology.n_atoms
