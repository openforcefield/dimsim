import typing

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
