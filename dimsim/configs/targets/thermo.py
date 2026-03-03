import typing

import datasets
import pyarrow

from dimsim.datasets.phase import PropertyPhase
from dimsim.molecule import map_smiles

EntryTag = typing.Literal[
    "density",
    "excess_molar_volume",
    "dielectric_constant",
    "enthalpy_of_mixing",
    "enthalpy_of_vaporization",
    "vapor_pressure",
]
"""

DATA_SCHEMA = pyarrow.schema(
    [
        ("type", pyarrow.string()),
        ("smiles_a", pyarrow.string()),
        ("x_a", pyarrow.float64()),
        ("smiles_b", pyarrow.string()),
        ("x_b", pyarrow.float64()),
        ("temperature", pyarrow.float64()),
        ("pressure", pyarrow.float64()),
        ("value", pyarrow.float64()),
        ("std", pyarrow.float64()),
        ("units", pyarrow.string()),
        ("source", pyarrow.string()),
    ]
)
"""
DATA_SCHEMA = pyarrow.schema(
    [
        ("phases", pyarrow.list_(pyarrow.string())),
        ("smiles", pyarrow.list_(pyarrow.string())),
        ("x", pyarrow.list_(pyarrow.float64())),
        ("temperature", pyarrow.float64()),
        ("pressure", pyarrow.float64()),
        ("value", pyarrow.float64()),
        ("std", pyarrow.float64()),
        ("units", pyarrow.string()),
        ("source", pyarrow.string()),
    ]
)


class DataEntry(typing.TypedDict):
    tag: typing.Literal[EntryTag]

    phases: PropertyPhase

    smiles: list[str]

    x: list[float]

    temperature: float

    pressure: float

    value: float

    std: float | None

    units: str

    source: str


class DensityEntry(DataEntry):
    tag = "density"

    phases: PropertyPhase = PropertyPhase.Liquid


class ExcessMolarVolumeEntry(DataEntry):
    tag = "excess_molar_volume"

    phases: PropertyPhase = PropertyPhase.Liquid


class DielectricConstantEntry(DataEntry):
    tag = "dielectric_constant"

    phases: PropertyPhase = PropertyPhase.Liquid


class EnthalpyOfMixingEntry(DataEntry):
    tag = "enthalpy_of_mixing"

    phases: PropertyPhase = PropertyPhase.Liquid


class EnthalpyOfVaporizationEntry(DataEntry):
    tag = "enthalpy_of_vaporization"

    phases: PropertyPhase = PropertyPhase.Liquid | PropertyPhase.Gas


class VaporPressureEntry(DataEntry):
    tag = "vapor_pressure"

    phases: PropertyPhase = PropertyPhase.Liquid | PropertyPhase.Gas


def create_dataset(*rows: DataEntry) -> datasets.Dataset:
    for row in rows:
        row["smiles"] = [map_smiles(value) for value in row["smiles"]]

    # TODO: validate rows
    table = pyarrow.Table.from_pylist([*rows], schema=DATA_SCHEMA)

    return datasets.Dataset(datasets.table.InMemoryTable(table))
