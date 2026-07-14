"""Prep (simulation) jobs from (thermophysical) data entries."""

from collections.abc import Sequence

from dimsim.configs._compute import BaseComputeConfig
from dimsim.configs.liquid import BulkLiquid


def compute_configs_from_data_entries(
    data_entries: list[dict],
    force_field: str,
    n_molecules: int,
) -> Sequence[BaseComputeConfig]:
    """Convert a list of thermophysical data entries into a list of simulation configs."""
    compute_configs: list[BaseComputeConfig] = list()

    for data_entry in data_entries:
        these_configs = _compute_configs_from_data_entry(data_entry, force_field, n_molecules)
        compute_configs.extend(these_configs)

    return compute_configs


def _compute_configs_from_data_entry(
    data_entry: dict,
    force_field: str,
    n_molecules: int,
) -> Sequence[BaseComputeConfig]:
    """Convert a single thermophysical data entry into a list of simulation configs."""
    match data_entry:
        case {"tag": "density" | "dielectric_constant"}:
            return _make_liquid_density_compute_configs(data_entry, force_field, n_molecules)
        case {"tag": "enthalpy_of_mixing" | "exceess_molar_volume"}:
            return _make_enthalpy_of_mixing_compute_configs(data_entry, force_field, n_molecules)
        case {"tag": "enthalpy_of_vaporization"}:
            return _make_enthalpy_of_vaporization_compute_configs(data_entry, force_field, n_molecules)
        case _:
            print(1)
            raise ValueError(f"Unsupported data entry tag: {data_entry['tag']}")


def _make_liquid_density_compute_configs(
    data_entry: dict,
    force_field: str,
    n_molecules: int,
) -> Sequence[BaseComputeConfig]:
    from dimsim.configs.liquid import BulkLiquid

    return tuple([
        BulkLiquid(
            tag="liquid",
            force_field=force_field,
            n_molecules=n_molecules,
            smiles=data_entry["smiles"],
            x=data_entry["x"],
            temperature=data_entry["temperature"],
            pressure=data_entry["pressure"],
            density=data_entry["value"],
        )
    ])


def _make_enthalpy_of_mixing_compute_configs(
    data_entry: dict,
    force_field: str,
    n_molecules: int,
) -> Sequence[BulkLiquid]:
    from dimsim.configs.liquid import BulkLiquid

    liquid_configs = [
        BulkLiquid(
            tag="liquid",
            force_field=force_field,
            n_molecules=n_molecules,
            smiles=data_entry["smiles"],
            x=data_entry["x"],
            temperature=data_entry["temperature"],
            pressure=data_entry["pressure"],
            density=None,
        )
    ]

    for component_smiles, component_x in zip(data_entry["smiles"], data_entry["x"]):
        liquid_configs.append(
            BulkLiquid(
                tag="liquid",
                force_field=force_field,
                n_molecules=n_molecules,  # not sure if each pure simulation should have the full n_molecules?
                smiles=[component_smiles],
                x=[1.0],
                temperature=data_entry["temperature"],
                pressure=data_entry["pressure"],
                density=None,
            )
        )

    return tuple(liquid_configs)


def _make_enthalpy_of_vaporization_compute_configs(
    data_entry: dict,
    force_field: str,
    n_molecules: int,
) -> Sequence[BaseComputeConfig]:
    from dimsim.configs.gas import VacuumGas
    from dimsim.configs.liquid import BulkLiquid

    return (
        BulkLiquid(
            tag="liquid",
            force_field=force_field,
            n_molecules=n_molecules,
            smiles=data_entry["smiles"],
            x=data_entry["x"],
            temperature=data_entry["temperature"],
            pressure=data_entry["pressure"],
            density=None,
        ),
        VacuumGas(
            tag="gas",
            force_field=force_field,
            n_molecules=n_molecules,
            smiles=data_entry["smiles"],
            x=data_entry["x"],
            temperature=data_entry["temperature"],
        ),
    )
