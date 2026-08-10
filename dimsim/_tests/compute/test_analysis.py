import shutil
from importlib.resources import files

from parsl import File

from dimsim.compute._analyze import _run_density_analysis
from dimsim.compute._files import ProductionFiles


def test_basic_analysis(tmp_path):
    # shim - see comment in source code
    for file in ["production.csv"]:
        shutil.copy(
            str(files("dimsim") / f"_tests/data/app_files/sample_density/{file}"),
            str(tmp_path / file),
        )

    production_files = ProductionFiles(
        topology=File(""),
        dcd_trajectory=File(""),
        msgpack_trajectory=File(""),
        log=File(""),
        state_data=File(str(tmp_path / "production.csv")),
        system=File(""),
        integrator=File(""),
        checkpoint=File(""),
    )

    analysis_result = _run_density_analysis(
        production_future={"simulation_files": production_files},
        job_dir=str(tmp_path),
    )

    # just some non-zero number
    assert analysis_result["mean"] > 0

    # std should be <10% of the mean
    assert analysis_result["std"] / analysis_result["mean"] < 0.1
