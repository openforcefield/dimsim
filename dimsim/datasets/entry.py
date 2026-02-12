import typing

EntryType = typing.Literal[
    "density",
    "dhvap",  # enthalpy_of_vaporization
    "dhmix",  # enthalpy_of_mixing
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

    # could define a default unit, like Evaluator does
    #
    # >>> from openff.evaluator.properties import EnthalpyOfMixing
    # >>> EnthalpyOfMixing.default_unit()
    # <Unit('kilojoule / mole')>

    id: str
