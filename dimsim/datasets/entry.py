import typing

EntryType = typing.Literal[
    "density",
    "hvap",  # enthalpy_of_vaporization
    "hmix",  # enthalpy_of_mixing
    "dielectric_constant",
    "osmotic_coefficient",
    "solvation_free_energy",
]


class DataEntry(typing.TypedDict):

    tag: EntryType

    smiles: list[str]

    x: list[float]

    temperature: float

    pressure: float

    value: float

    std: float

    units: str

    source: str
