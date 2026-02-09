import numpy as np
import pytest
from openff.toolkit import Molecule

from dimsim._tests.utils import get_test_data_path
from dimsim.datasets.thermoml import (
    thermoml_dataset_from_doi,
    thermoml_dataset_from_xml,
)


@pytest.mark.skip(reason="Not implemented yet")
@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            "single_density.xml",
            {
                "type": "density",
                "x_a": 1.0,
                "x_b": None,
                "temperature": 293.15,
                "pressure": 1.0,
                "value": 0.96488,
                "std": 0.05,
                "units": "g/mL",
                "source": "",
            },
        ),
        (
            "single_dhmix.xml",
            {
                "type": "dhmix",
                "x_a": 0.219,
                "x_b": 0.781,
                "temperature": 298.15,
                "pressure": 0.997,
                "value": 0.03021,
                "std": 0.000151,
                "units": "kcal/mol",
                "source": "10.1016/j.jct.2008.12.004",
            },
        ),
        (
            "single_dhvap.xml",
            {
                "type": "dhvap",
                "x_a": 1.0,
                "x_b": None,
                "temperature": 298.15,
                "pressure": None,
                "value": 10.51625,
                "std": 0.1434,
                "units": "kcal/mol",
                "source": "10.1016/j.fluid.2014.12.023",
            },
        ),
        (
            "single_dielectric.xml",
            {
                "type": "dielectric_constant",
                "x_a": 1.0,
                "x_b": None,
                "temperature": 298.15,
                "pressure": 0.997,
                "value": 11.76,
                "std": 0.02,
                "units": "dimensionless",
                "source": "",
            },
        ),
    ],
)
def test_load_property_types(filename: str, expected: dict):
    """Test loading a single data type from a ThermoML XML file"""
    dataset = thermoml_dataset_from_xml(get_test_data_path(f"thermoml/{filename}"))
    assert len(dataset) == 1

    entry = dataset[0]
    assert entry["type"] == expected["type"]
    assert entry["x_a"] == expected["x_a"]

    # assert mapped smiles
    Molecule.from_mapped_smiles(entry["smiles_a"])
    if expected["x_b"] is not None:
        assert entry["x_b"] == expected["x_b"]
        Molecule.from_mapped_smiles(entry["smiles_b"])
    else:
        assert entry["smiles_b"] is None
        assert entry["x_b"] is None

    assert entry["temperature"] == expected["temperature"]
    if expected["pressure"] is not None:
        assert np.isclose(entry["pressure"], expected["pressure"], atol=1e-3)
    else:
        assert entry["pressure"] is None

    assert np.isclose(entry["value"], expected["value"], atol=1e-5)
    assert np.isclose(entry["std"], expected["std"], atol=1e-5)
    assert entry["units"] == expected["units"]

    assert entry["source"] == expected["source"]


@pytest.mark.skip(reason="Not implemented yet")
def test_load_single_osmotic():
    """
    Test loading a single osmotic coefficient data point from a ThermoML XML file.

    This is analogous to the test above,
    but is included here to ensure that ions are dealt with correctly.

    """
    dataset = thermoml_dataset_from_xml(get_test_data_path("thermoml/single_osmotic.xml"))
    assert len(dataset) == 1

    entry = dataset[0]
    assert entry["type"] == "osmotic_coefficient"

    assert "." in entry["smiles_a"]
    Molecule.from_mapped_smiles(entry["smiles_a"])
    assert entry["x_a"] == 0.00086

    Molecule.from_mapped_smiles(entry["smiles_b"])
    assert entry["x_b"] == 0.99914

    assert np.isclose(entry["temperature"], 298.15, atol=1e-3)
    assert entry["pressure"] is None
    assert np.isclose(entry["value"], 0.7389, atol=1e-5)
    assert np.isclose(entry["std"], 0.00655, atol=1e-5)
    assert entry["units"] == "dimensionless"
    assert entry["source"] == "10.1016/j.fluid.2006.09.025"


@pytest.mark.skip(reason="Not implemented yet")
def test_load_from_doi():
    """Test loading a ThermoML dataset from a DOI"""
    dataset = thermoml_dataset_from_doi("10.1016/j.fluid.2014.12.023")
    assert len(dataset) == 9
    for entry in dataset:
        assert entry["source"] == "10.1016/j.fluid.2014.12.023"
