import numpy
import pytest
from openff.toolkit import Molecule

from dimsim._tests.utils import get_test_data_path
from dimsim.datasets.thermoml.thermoml import ThermoMLDataSet


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            "single_density.xml",
            {
                "tag": "density",
                "x": [1.0],
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
                "tag": "dhmix",
                "x": [0.219, 0.781],
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
                "tag": "dhvap",
                "x": [1.0],
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
                "tag": "dielectric_constant",
                "x": [1.0],
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
    dataset = ThermoMLDataSet.from_xml(open(get_test_data_path(f"thermoml/{filename}")).read())
    assert len(dataset) == 1

    entry = next(iter(dataset))
    assert entry["tag"] == expected["tag"]
    assert entry["x"] == expected["x"]

    assert len(entry["x"]) == len(entry["smiles"])
    assert len(entry["x"]) == len(expected["x"])

    for found_x, expected_x, found_smiles in zip(
        entry["x"],
        expected["x"],
        entry["smiles"],
    ):
        assert found_x == expected_x

        # just make sure it's valid SMILES
        Molecule.from_smiles(found_smiles)

        # Evaluator uses non-mapped SMILES, pseudocode here used mapped
        # Molecule.from_mapped_smiles(found_smiles)

    assert entry["temperature"] == expected["temperature"]
    if expected["pressure"] is not None:
        assert numpy.isclose(entry["pressure"], expected["pressure"], atol=1e-3)
    else:
        assert entry["pressure"] is None

    assert numpy.isclose(entry["value"], expected["value"], atol=1e-5)
    assert numpy.isclose(entry["std"], expected["std"], atol=1e-5)
    assert entry["units"] == expected["units"]

    assert entry["source"] == expected["source"]


def test_load_single_osmotic():
    """
    Test loading a single osmotic coefficient data point from a ThermoML XML file.

    This is analogous to the test above,
    but is included here to ensure that ions are dealt with correctly.

    """
    dataset = ThermoMLDataSet.from_xml(open(get_test_data_path("thermoml/single_osmotic.xml")).read())
    assert len(dataset) == 1

    entry = next(iter(dataset))
    assert entry["tag"] == "osmotic_coefficient"

    assert "." in entry["smiles"]
    Molecule.from_mapped_smiles(entry["smiles"])
    assert entry["x"] == 0.00086

    Molecule.from_mapped_smiles(entry["smiles"])
    assert entry["x"] == 0.99914

    assert numpy.isclose(entry["temperature"], 298.15, atol=1e-3)
    assert entry["pressure"] is None
    assert numpy.isclose(entry["value"], 0.7389, atol=1e-5)
    assert numpy.isclose(entry["std"], 0.00655, atol=1e-5)
    assert entry["units"] == "dimensionless"
    assert entry["source"] == "10.1016/j.fluid.2006.09.025"


def test_load_from_doi():
    """Test loading a ThermoML dataset from a DOI"""
    dataset = ThermoMLDataSet.from_doi("10.1016/j.fluid.2014.12.023")
    assert len(dataset) == 9
    for entry in dataset:
        assert entry["source"] == "10.1016/j.fluid.2014.12.023"
