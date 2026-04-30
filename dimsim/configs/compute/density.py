import typing

from dimsim.configs.targets.thermo import DataEntry


class DensityConfig(typing.TypedDict):
    tag: typing.Literal["density"] = "density"

    target: DataEntry

    force_field: str

    n_molecules: int

    """
    Maybe add an RNG seed?
    """
