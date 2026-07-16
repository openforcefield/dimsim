import pytest
from openff.toolkit import Topology

from dimsim.compute.apps import prepare_packed_topology
from dimsim.configs.liquid import BulkLiquid
from dimsim.configs.targets.thermo import DataEntry


@pytest.fixture
def sample_density_target() -> DataEntry:
    return DataEntry(
        id=99595555472285796331286665565837685644847269654436554750580343052999515028260,
        tag="density",
        smiles=["COCCO"],
        x=[1.0],
        temperature=293.15,
        pressure=101.3,
        value=0.96488,
        std=5.0e-05,
        units="gram / milliliter",
        source="",
    )


@pytest.fixture
def sample_bulk_liquid_config(sample_density_target) -> BulkLiquid:
    return BulkLiquid(
        tag="liquid",
        force_field="test.offxml",
        n_molecules=1000,
        smiles=["COCCO"],
        x=[1.0],
        temperature=293.15,
        pressure=101.3,
        value=0.96488,
    )


"""
@python_app
def prepare_packed_topology(
    compute_config: BulkLiquid,
    job_dir: str,
) -> dict[str, BulkLiquid | Topology]:
"""


class TestPreparePackedTopology:
    def test_prepare_packed_topology(self, sample_bulk_liquid_config, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        result = prepare_packed_topology(sample_bulk_liquid_config, str(job_dir)).result()

        assert "compute_config" in result
        assert "packed_topology" in result
        assert isinstance(result["compute_config"], dict)
        assert result["compute_config"]["tag"] == "liquid"
        assert isinstance(result["packed_topology"], Topology)
