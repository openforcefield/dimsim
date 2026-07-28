import json
import shutil
from importlib.resources import files

import openmm
import pytest
from openff.toolkit import Topology

from dimsim.compute._minimize import _minimize_energy
from dimsim.configs.liquid import BulkLiquid


@pytest.fixture
def prepare_future() -> dict[str, BulkLiquid | Topology]:
    compute_config = BulkLiquid(
        json.load(
            open(
                files("dimsim") / "_tests/data/app_files/sample_density/compute_config.json",
            ),
        ),
    )
    return {
        "compute_config": compute_config,
        "openmm_system": openmm.XmlSerializer.deserialize(
            open(
                files("dimsim") / "_tests/data/app_files/sample_density/openmm_system.xml",
            ).read(),
        ),
    }


def test_minimize_basic(prepare_future, tmp_path):
    # shim - see comment in source code
    shutil.copy(
        str(files("dimsim") / "_tests/data/app_files/sample_density/packed_topology.pdb"),
        str(tmp_path / "packed_topology.pdb"),
    )

    minimize_result = _minimize_energy(
        system_future=prepare_future,
        job_dir=str(tmp_path),
    )

    assert minimize_result["final"] < minimize_result["original"]
