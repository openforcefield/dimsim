import uuid

import numpy
import pytest
from openff.toolkit import Molecule
from openff.units import Unit

from dimsim._tests.utils import get_test_data_path
from dimsim.datasets.thermoml import ThermoMLDataSet


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            "single_density.xml",
            {
                "x": [1.0],
                "temperature": 293.15,
                "pressure": 1.0,
                "value": 0.96488,
                "std": 0.00005,
                "units": "g/mL",
                "source": "",
            },
        ),
        (
            "single_dhmix.xml",
            {
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
                "x": [1.0],
                "temperature": 293.15,
                "pressure": 0.997,
                "value": 11.76,
                "std": 0.02,
                "units": "dimensionless",
                "source": "",
            },
        ),
    ],
)
class TestThermoMLDataset:
    """Class set up only to make convenient re-use of parametrized test cases."""

    def test_load_property_types(self, filename: str, expected: dict):
        """Test loading a single data type from a ThermoML XML file"""
        dataset = ThermoMLDataSet.from_xml(open(get_test_data_path(f"thermoml/{filename}")).read())
        assert len(dataset) == 1

        entry = next(iter(dataset))
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
        assert Unit(entry["units"]) == Unit(expected["units"])

        assert entry["source"] == expected["source"]

    def test_same_property_same_hash(self, filename, expected):
        dataset = ThermoMLDataSet.from_xml(open(get_test_data_path(f"thermoml/{filename}")).read())
        this_property = next(iter(dataset))

        reloaded = ThermoMLDataSet.from_xml(open(get_test_data_path(f"thermoml/{filename}")).read())
        reloaded_property = next(iter(reloaded))

        assert reloaded_property["id"] == this_property["id"]

    def test_different_property_different_hash(self, filename, expected):
        dataset = ThermoMLDataSet.from_xml(open(get_test_data_path(f"thermoml/{filename}")).read())
        this_property = next(iter(dataset))

        # Grab a different property from a different file, doesn't really matter which one it is
        other_filename = {
            "single_density.xml": "single_dhmix.xml",
            "single_dhmix.xml": "single_dhvap.xml",
            "single_dhvap.xml": "single_dielectric.xml",
            "single_dielectric.xml": "single_density.xml",
        }[filename]

        different_dataset = ThermoMLDataSet.from_xml(open(get_test_data_path(f"thermoml/{other_filename}")).read())
        different_property = next(iter(different_dataset))

        assert different_property["id"] != this_property["id"]

    def test_pandas_roundtrip(self, filename, expected):
        dataset = ThermoMLDataSet.from_xml(open(get_test_data_path(f"thermoml/{filename}")).read())

        roundtripped = ThermoMLDataSet.from_pandas(dataset.to_pandas())

        assert len(dataset) == len(roundtripped)

        for property1, property2 in zip(dataset, roundtripped):
            assert property1.keys() == property2.keys()
            for key in property1.keys():
                assert property1[key] == property2[key]


@pytest.mark.skip(reason="Implement next")
def test_load_single_osmotic():
    """
    Test loading a single osmotic coefficient data point from a ThermoML XML file.

    This is analogous to the test above,
    but is included here to ensure that ions are dealt with correctly.

    """
    dataset = ThermoMLDataSet.from_xml(open(get_test_data_path("thermoml/single_osmotic.xml")).read())
    assert len(dataset) == 1

    entry = next(iter(dataset))

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


@pytest.mark.skip(reason="implement next")
def test_load_from_doi():
    """Test loading a ThermoML dataset from a DOI"""
    dataset = ThermoMLDataSet.from_doi("10.1016/j.fluid.2014.12.023")
    assert len(dataset) == 186
    for entry in dataset:
        assert entry["source"] == "10.1016/j.fluid.2014.12.023"


def test_to_pandas():
    """A test to ensure that data sets are convertable to pandas objects."""

    thermoml_dataset = ThermoMLDataSet()

    density_entry = {
        "id": str(uuid.uuid4()).replace("-", ""),
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

    thermoml_dataset.add_properties(density_entry)

    dataframe = thermoml_dataset.to_pandas()

    required_columns = [
        "Id",
        "tag",
        "Temperature (K)",
        "Pressure (kPa)",
        "N Components",
        "Component 1",
        "Mole Fraction 1",
        "Value",
        "Uncertainty",
        "Source",
    ]

    assert all(x in dataframe for x in required_columns)

    assert dataframe is not None
    assert dataframe.shape == (1, 10)

    # Source may be an empty string but is not NaN - is this behavior okay?
    data_set_without_na = dataframe.dropna(axis=1, how="all")
    assert data_set_without_na.shape == (1, 10)
