"""Test configuration and fixtures."""

import random

import pytest

from dimsim.configs.targets.thermo import DataEntry


@pytest.fixture
def density_entry() -> DataEntry:
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
def dielectric_entry() -> DataEntry:
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
