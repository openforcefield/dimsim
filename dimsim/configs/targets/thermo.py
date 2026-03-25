import typing

import datasets
import pyarrow

from dimsim.molecule import map_smiles

DATA_SCHEMA = pyarrow.schema(
    [
        ("id", pyarrow.string()),
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
    id: str

    tag: str  # was previously EntryTag

    smiles: list[str]

    x: list[float]

    temperature: float

    pressure: float

    value: float

    std: float | None

    units: str

    source: str


def create_pyarrow_dataset(rows: typing.Iterable[DataEntry]) -> datasets.Dataset:
    for row in rows:
        row["smiles"] = [map_smiles(value) for value in row["smiles"]]

    table = pyarrow.Table.from_pylist([*rows], schema=DATA_SCHEMA)

    return datasets.Dataset(datasets.table.InMemoryTable(table))
