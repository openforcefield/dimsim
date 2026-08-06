import shutil
from importlib.resources import files

import pytest
from parsl import File

from dimsim.compute._files import PreparingFiles
from dimsim.compute._minimize import _minimize_energy


@pytest.fixture
def prepare_future() -> dict[str, PreparingFiles]:
    return {
        "prepared_files": PreparingFiles(
            openmm_system=File(files("dimsim") / "_tests/data/app_files/sample_density/openmm_system.xml"),
        ),
    }


def test_minimize_basic(prepare_future, tmp_path):
    # shim - see comment in source code
    for file in ["compute_config.json", "packed_topology.pdb", "openmm_system.xml"]:
        shutil.copy(
            str(files("dimsim") / f"_tests/data/app_files/sample_density/{file}"),
            str(tmp_path / file),
        )

    minimize_result = _minimize_energy(
        system_future=prepare_future,
        job_dir=str(tmp_path),
    )

    assert minimize_result["final"] < minimize_result["original"]
