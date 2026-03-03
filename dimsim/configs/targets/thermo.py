import typing

import datasets
import pyarrow

from dimsim.datasets.datasets import PropertyPhase
from dimsim.molecule import map_smiles

EntryTag = typing.Literal["density"]

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
    phases: list[PropertyPhase]

    smiles: list[str]

    x: list[float]

    temperature: float

    pressure: float

    value: float

    std: float | None

    units: str

    source: str


class DensityEntry(DataEntry):
    phases: list[PropertyPhase] = PropertyPhase.Liquid


class ExcessMolarVolumeEntry(DataEntry):
    phases: list[PropertyPhase] = PropertyPhase.Liquid


class DielectricConstantEntry(DataEntry):
    phases: list[PropertyPhase] = PropertyPhase.Liquid


class EnthalpyOfMixingEntry(DataEntry):
    phases: list[PropertyPhase] = PropertyPhase.Liquid


class EnthalpyOfVaporizationEntry(DataEntry):
    phases: list[PropertyPhase] = PropertyPhase.Liquid | PropertyPhase.Gas


class VaporPressureEntry(DataEntry):
    phases: list[PropertyPhase] = PropertyPhase.Liquid | PropertyPhase.Gas


def create_dataset(*rows: DataEntry) -> datasets.Dataset:
    for row in rows:
        row["smiles"] = [map_smiles(value) for value in row["smiles"]]

    # TODO: validate rows
    table = pyarrow.Table.from_pylist([*rows], schema=DATA_SCHEMA)

    return datasets.Dataset(datasets.table.InMemoryTable(table))
