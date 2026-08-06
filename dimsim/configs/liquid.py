import typing

import pyarrow

from dimsim.configs._compute import BaseComputeConfig
from dimsim.configs.targets.thermo import DataEntry


class BulkLiquid(BaseComputeConfig):
    tag: typing.Literal["liquid"]

    """
    Maybe add an RNG seed?
    """

    pressure: float

    density: float | None

    """
    maybe?

    units: str
    """


# also has
"""
    force_field: str

    n_molecules: int

    smiles: list[str]

    x: list[float]

    temperature: float
"""


def liquid_config_from_data_entry(
    data_entry: DataEntry,
    force_field: str,
    n_molecules: int,
) -> BulkLiquid:
    """
    Create a `BulkLiquid` config from thermophysical and chemical information in a `DataEntry`
    and job-specific inputs.
    """
    return BulkLiquid(
        tag="liquid",
        force_field=force_field,
        n_molecules=n_molecules,
        smiles=data_entry["smiles"],
        x=data_entry["x"],
        temperature=data_entry["temperature"],
        pressure=data_entry["pressure"],
        density=data_entry["value"],
    )


LIQUID_SCHEMA = pyarrow.schema(
    [
        ("tag", pyarrow.string()),
        ("force_field", pyarrow.string()),
        ("n_molecules", pyarrow.int64()),
        ("smiles", pyarrow.list_(pyarrow.string())),
        ("x", pyarrow.list_(pyarrow.float64())),
        ("temperature", pyarrow.float64()),
        ("pressure", pyarrow.float64()),
        ("density", pyarrow.float64()),
    ]
)


def make_liquid_table(entries: list[BulkLiquid], job_id: str | None = None) -> pyarrow.Table:
    """
    Build a Table from DataEntry dicts. If `job_id` is given, it's applied
    to every row (the normal case: one task -> one job_id -> N entries).
    """
    columns: dict[str, list] = {name: [e[name] for e in entries] for name in BulkLiquid.__annotations__}
    columns["job_id"] = [job_id] * len(entries)
    return pyarrow.Table.from_pydict(columns, schema=LIQUID_SCHEMA)
