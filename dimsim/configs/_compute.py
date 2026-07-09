import typing

import typing_extensions  # PEP 728 backport for Python 3.12 - drop when 3.13+


class BaseComputeConfig(typing_extensions.TypedDict, extra_items=typing.Any):  # type: ignore[call-arg]
    # mypy does not yet support extra_items
    # https://github.com/python/mypy/issues/18176
    force_field: str

    n_molecules: int

    smiles: list[str]

    x: list[float]

    temperature: float
