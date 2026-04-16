import typing

from dimsim.configs.targets.thermo import DataEntry


class DensityConfig(typing.TypedDict):
    target: DataEntry

    force_field: str

    n_molecules: int
