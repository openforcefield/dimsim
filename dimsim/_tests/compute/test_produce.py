import shutil
from importlib.resources import files

import pytest
from parsl import File

from dimsim.compute._files import EquilibrationFiles
from dimsim.compute._produce import _run_production


@pytest.fixture
def equilibration_future() -> dict[str, EquilibrationFiles]:
    return {
        "simulation_files": EquilibrationFiles(
            topology=File(files("dimsim") / "_tests/data/app_files/sample_density/equilibrated_topology.pdb"),
            dcd_trajectory=File("foo.dcd"),
            msgpack_trajectory=File("foo.msgpack"),
            log=File(files("dimsim") / "_tests/data/app_files/sample_density/equilibrate.log"),
            state_data=File(files("dimsim") / "_tests/data/app_files/sample_density/equilibration.csv"),
            system=File(files("dimsim") / "_tests/data/app_files/sample_density/equilibration_system.xml"),
            integrator=File(files("dimsim") / "_tests/data/app_files/sample_density/equilibration_integrator.xml"),
            checkpoint=File(files("dimsim") / "_tests/data/app_files/sample_density/equilibration_checkpoint.chk"),
        )
    }


def test_basic(tmp_path, equilibration_future):
    for file in ["compute_config.json", "packed_topology.pdb", "openmm_system.xml"]:
        shutil.copy(
            str(files("dimsim") / f"_tests/data/app_files/sample_density/{file}"),
            str(tmp_path / file),
        )
    pass

    _run_production(
        production_config=None,  # Replace with an actual ProductionConfig object
        equilibration_future=equilibration_future,
        job_dir=str(tmp_path),
    )
