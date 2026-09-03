import shutil
from importlib.resources import files

import pytest
from parsl import File

from dimsim.compute._equilibrate import _run_equilibration
from dimsim.compute._files import MinimizationFiles


@pytest.fixture
def minimization_future() -> dict[str, MinimizationFiles]:
    return {
        "simulation_files": MinimizationFiles(
            topology=File(files("dimsim") / "_tests/data/app_files/sample_density/minimized_topology.pdb"),
            system=File(files("dimsim") / "_tests/data/app_files/sample_density/minimized_system.xml"),
            integrator=File(files("dimsim") / "_tests/data/app_files/sample_density/minimized_integrator.xml"),
            checkpoint=File(files("dimsim") / "_tests/data/app_files/sample_density/minimized_checkpoint.chk"),
        )
    }


def test_basic(tmp_path, minimization_future):
    for file in ["compute_config.json", "packed_topology.pdb", "openmm_system.xml"]:
        shutil.copy(
            str(files("dimsim") / f"_tests/data/app_files/sample_density/{file}"),
            str(tmp_path / file),
        )
    pass

    _run_equilibration(
        equilibration_config=None,  # Replace with an actual EquilibrationConfig object
        minimization_future=minimization_future,
        job_dir=str(tmp_path),
    )
