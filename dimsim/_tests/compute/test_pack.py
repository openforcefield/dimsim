import pytest
from openff.toolkit import Molecule, Topology
from parsl import File

from dimsim.compute._pack import _prepare_packed_topology
from dimsim.configs.liquid import BulkLiquid


@pytest.fixture
def bulk_liquid() -> BulkLiquid:
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


def _get_density(topology: Topology) -> float:
    # even with non-cubic boxes, volume is supposed to be a_x * b_y * c_z, so this should be fine
    # "The volume of the unit cell " ...
    # https://docs.openmm.org/latest/userguide/theory/05_other_features.html
    return (
        sum(atom.mass for atom in topology.atoms)
        / (topology.box_vectors[0, 0] * topology.box_vectors[1, 1] * topology.box_vectors[2, 2])
    ).m_as("g/mL")


def test_basic_packing(bulk_liquid, tmp_path):
    result = _prepare_packed_topology(bulk_liquid, str(tmp_path))

    assert isinstance(result["packed_files"], dict)
    assert isinstance(result["packed_files"]["packed_topology"], File)

    topology = Topology.from_pdb(
        file_path=result["packed_files"]["packed_topology"].filepath,
        unique_molecules=[Molecule.from_smiles(smiles) for smiles in result["compute_config"]["smiles"]],
    )
    assert topology.n_molecules == bulk_liquid["n_molecules"]

    # currently the default is to scale to 0.7 * density (if defined)
    assert _get_density(topology) == pytest.approx(bulk_liquid["density"] * 0.7, rel=0.1)


def test_packing_with_no_density_in_target(bulk_liquid, tmp_path):
    """When density is None, test that the result is packed to 0.7 g/mL"""
    bulk_liquid_no_density = bulk_liquid.copy()
    del bulk_liquid_no_density["density"]

    result = _prepare_packed_topology(bulk_liquid_no_density, str(tmp_path))

    assert isinstance(result["packed_files"], dict)
    assert isinstance(result["packed_files"]["packed_topology"], File)

    topology = Topology.from_pdb(
        file_path=result["packed_files"]["packed_topology"].filepath,
        unique_molecules=[Molecule.from_smiles(smiles) for smiles in result["compute_config"]["smiles"]],
    )
    assert topology.n_molecules == bulk_liquid_no_density["n_molecules"]

    assert _get_density(topology) == pytest.approx(0.7, rel=0.1)


def test_packing_with_altered_n_molecules(bulk_liquid, tmp_path):
    """Test that the result is packed to the correct number of molecules"""
    bulk_liquid_altered_n_molecules = bulk_liquid.copy()
    bulk_liquid_altered_n_molecules["n_molecules"] = 210

    result = _prepare_packed_topology(bulk_liquid_altered_n_molecules, str(tmp_path))

    assert isinstance(result["packed_files"], dict)
    assert isinstance(result["packed_files"]["packed_topology"], File)

    topology = Topology.from_pdb(
        file_path=result["packed_files"]["packed_topology"].filepath,
        unique_molecules=[Molecule.from_smiles(smiles) for smiles in result["compute_config"]["smiles"]],
    )
    assert topology.n_molecules == bulk_liquid_altered_n_molecules["n_molecules"]

    # currently the default is to scale to 0.7 * density (if defined)
    assert _get_density(topology) == pytest.approx(bulk_liquid_altered_n_molecules["density"] * 0.7, rel=0.1)
