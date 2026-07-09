import typing


class BaseComputeConfig(typing.TypedDict):
    force_field: str

    n_molecules: int

    smiles: list[str]

    x: list[float]

    temperature: float
