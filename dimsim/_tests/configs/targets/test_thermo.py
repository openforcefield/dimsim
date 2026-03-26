import random

import pytest
from openff.toolkit import Molecule

from dimsim.configs.targets.thermo import DataEntry, create_dataset


def create_test_entry() -> DataEntry:
    n_components = random.randint(1, 4)

    STOCK_SMILES = [
        "C",
        "CCO",
        "C(Cl)(Cl)(Cl)Cl",
        "c1ccccc1",
        "CC(=O)O",
    ]

    x = [random.random() for _ in range(n_components)]
    x = [value / sum(x) for value in x]

    return DataEntry(
        id="test",
        tag="test",
        smiles=random.sample(STOCK_SMILES, n_components),
        x=x,
        temperature=random.uniform(210.0, 450.0),
        pressure=1.0,
        value=random.uniform(0.1, 2.0),
        std=random.uniform(0.001, 0.05),
        units="g/mL",
        source="",
    )


def test_create_dataset():
    entries = [create_test_entry() for _ in range(5)]

    dataset = create_dataset(entries)
    assert len(dataset) == 5

    for row, entry in zip(dataset, entries):
        assert row["id"] == entry["id"]
        assert row["tag"] == entry["tag"]

        # These should not be equal, at least in this test,
        # since the test data is not mapped, but the stored data is mapped
        # Revisit if DataEntry is changed to store mapped SMILES
        assert row["smiles"] != entry["smiles"]

        # make sure the stored SMILES is mapped and matches the original SMILES
        for dataset_smiles, entry_smiles in zip(row["smiles"], entry["smiles"]):
            assert ":1" in dataset_smiles
            assert Molecule.from_mapped_smiles(dataset_smiles) == Molecule.from_smiles(entry_smiles)

        assert row["x"] == entry["x"]

        assert 1.0 == pytest.approx(sum(row["x"]))
        assert 1.0 == pytest.approx(sum(entry["x"]))

        assert row["temperature"] == entry["temperature"]
        assert row["pressure"] == entry["pressure"]
        assert row["value"] == entry["value"]
        assert row["std"] == entry["std"]
        assert row["units"] == entry["units"]
        assert row["source"] == entry["source"]
