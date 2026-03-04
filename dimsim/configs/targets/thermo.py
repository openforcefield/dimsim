import typing

import datasets
import pyarrow

from dimsim.molecule import map_smiles

EntryTag = typing.Literal[
    "density",
    "excess_molar_volume",
    "dielectric_constant",
    "enthalpy_of_mixing",
    "enthalpy_of_vaporization",
    "vapor_pressure",
]

DATA_SCHEMA = pyarrow.schema(
    [
        ("tag", pyarrow.string()),
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
    tag: str  # EntryTag

    smiles: list[str]

    x: list[float]

    temperature: float

    pressure: float

    value: float

    std: float | None

    units: str

    source: str


"""
class DensityEntry(DataEntry):
    tag: EntryTag = "density"

    phases: PropertyPhase = PropertyPhase.Liquid


class ExcessMolarVolumeEntry(DataEntry):
    tag: typing.Literal["excess_molar_volume"] = "excess_molar_volume"

    phases: PropertyPhase = PropertyPhase.Liquid


class DielectricConstantEntry(DataEntry):
    tag: typing.Literal["dielectric_constant"] = "dielectric_constant"

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
"""


def create_dataset(*rows: DataEntry) -> datasets.Dataset:
    for row in rows:
        row["smiles"] = [map_smiles(value) for value in row["smiles"]]

    # TODO: validate rows
    table = pyarrow.Table.from_pylist([*rows], schema=DATA_SCHEMA)

    return datasets.Dataset(datasets.table.InMemoryTable(table))
