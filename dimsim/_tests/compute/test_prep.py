import random

import pytest

from dimsim.compute.prep import compute_configs_from_data_entries, get_liquid_deduplication_key


@pytest.fixture
def density_entry():
    return {
        "id": random.randint(10**15, 10**16 - 1),  # random 16-digit integer
        "tag": "density",
        "x": [1.0],
        "smiles": ["[C:1]([O:5][C:3]([C:2]([O:4][H:13])([H:9])[H:10])([H:11])[H:12])([H:6])([H:7])[H:8]"],
        "temperature": 293.15,
        "pressure": 1.0,
        "value": 0.96488,
        "std": 0.00005,
        "units": "g/mL",
        "source": "",
    }


@pytest.fixture
def dielectric_entry():
    return {
        "id": random.randint(10**15, 10**16 - 1),  # random 16-digit integer
        "tag": "dielectric_constant",
        "x": [1.0],
        "smiles": ["[C:1]([O:5][C:3]([C:2]([O:4][H:13])([H:9])[H:10])([H:11])[H:12])([H:6])([H:7])[H:8]"],
        "temperature": 293.15,
        "pressure": 1.0,
        "value": 11.76,
        "std": 0.02,
        "units": "dimensionless",
        "source": "",
    }


def test_density_and_dielectric_produce_same_compute_config(density_entry, dielectric_entry):

    compute_configs = compute_configs_from_data_entries(
        data_entries=[density_entry, dielectric_entry],
        force_field="test-10.0",
        n_molecules=600,
    )

    assert len(set(get_liquid_deduplication_key(val) for val in compute_configs)) == 1


def test_basic_deduplciation(density_entry):
    """
    When data entry, force field, and n_molecules are the same, the resulting compute
    configs should "deduplicate" out, in the form of get_liquid_deduplication_key()
    returning identical values.
    """

    compute_configs = [
        *compute_configs_from_data_entries(
            data_entries=[density_entry],
            force_field="test-0.0a",
            n_molecules=600,
        ),
        *compute_configs_from_data_entries(
            data_entries=[density_entry],
            force_field="test-0.0a",
            n_molecules=600,
        ),
    ]

    assert len(set(get_liquid_deduplication_key(val) for val in compute_configs)) == 1


def test_different_force_fields_do_not_deduplicate(density_entry):

    compute_configs = [
        *compute_configs_from_data_entries(
            data_entries=[density_entry],
            force_field="test-10.0",
            n_molecules=600,
        ),
        *compute_configs_from_data_entries(
            data_entries=[density_entry],
            force_field="test-10.0.1",
            n_molecules=600,
        ),
    ]

    assert len(set(get_liquid_deduplication_key(val) for val in compute_configs)) == 2


def test_different_n_moelcules_do_not_deduplicate(density_entry):

    compute_configs = [
        *compute_configs_from_data_entries(
            data_entries=[density_entry],
            force_field="foo-bar.x",
            n_molecules=1000,
        ),
        *compute_configs_from_data_entries(
            data_entries=[density_entry],
            force_field="foo-bar.x",
            n_molecules=1001,
        ),
    ]

    assert len(set(get_liquid_deduplication_key(val) for val in compute_configs)) == 2
