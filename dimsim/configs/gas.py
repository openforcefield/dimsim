import typing

from dimsim.configs._compute import BaseComputeConfig
from dimsim.configs.targets.thermo import DataEntry


class VacuumGas(BaseComputeConfig):
    tag: typing.Literal["gas"]

    """
    Maybe add an RNG seed?
    """

    """
    maybe?

    units: str
    """


def gas_config_from_data_entry(
    data_entry: DataEntry,
    force_field: str,
    n_molecules: int,
) -> VacuumGas:
    """
    Create a `VacuumGas` config from thermophysical and chemical information in a `DataEntry`
    and job-specific inputs.
    """
    return VacuumGas(
        tag="gas",
        force_field=force_field,
        n_molecules=n_molecules,
        smiles=data_entry["smiles"],
        x=data_entry["x"],
        temperature=data_entry["temperature"],
        pressure=data_entry["pressure"],
        value=data_entry["value"],
    )
