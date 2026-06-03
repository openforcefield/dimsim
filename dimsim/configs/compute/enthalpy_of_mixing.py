import typing

from dimsim.configs.targets.thermo import DataEntry


class EnthalpyOfMixingConfig(typing.TypedDict):
    tag: typing.Literal["enthalpy_of_mixing"] = "enthalpy_of_mixing"

    target: DataEntry

    force_field: str

    n_molecules: int

    """
    Maybe add an RNG seed?
    """
