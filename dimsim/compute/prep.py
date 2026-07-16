"""Prep (simulation) jobs from (thermophysical) data entries."""

from collections.abc import Sequence

from dimsim.configs._compute import BaseComputeConfig
from dimsim.configs.gas import VacuumGas
from dimsim.configs.liquid import BulkLiquid
from dimsim.configs.targets.thermo import DataEntry


def compute_configs_from_data_entries(
    data_entries: list[DataEntry],
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
    data_entry: DataEntry,
    force_field: str,
    n_molecules: int,
) -> Sequence[BaseComputeConfig]:
    """Convert a single thermophysical data entry into a list of simulation configs."""
    match data_entry:
        case {"tag": "density" | "dielectric_constant"}:
            return _make_liquid_density_compute_configs(data_entry, force_field, n_molecules)
        case {"tag": "enthalpy_of_mixing" | "exceess_molar_volume"}:
            return _make_enthalpy_of_mixing_compute_configs(data_entry, force_field, n_molecules)  # type: ignore[arg-type]
        case {"tag": "enthalpy_of_vaporization"}:
            return _make_enthalpy_of_vaporization_compute_configs(data_entry, force_field, n_molecules)
        case _:
            raise ValueError(f"Unsupported data entry tag: {data_entry['tag']}")


def _make_liquid_density_compute_configs(
    data_entry: DataEntry,
    force_field: str,
    n_molecules: int,
) -> Sequence[BulkLiquid]:
    from dimsim.configs.liquid import BulkLiquid

    return tuple(
        [
            BulkLiquid(
                tag="liquid",
                force_field=force_field,
                n_molecules=n_molecules,
                smiles=data_entry["smiles"],
                x=data_entry["x"],
                temperature=data_entry["temperature"],
                pressure=data_entry["pressure"],
                density=data_entry["value"] if data_entry["tag"] == "density" else None,
            )
        ]
    )


def _make_enthalpy_of_mixing_compute_configs(
    data_entry: DataEntry,
    force_field: str,
    n_molecules: int,  # error if float? would this value ever be the result of rounding?
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
            density=data_entry["value"] if data_entry["tag"] == "density" else None,
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
                density=data_entry["value"] if data_entry["tag"] == "density" else None,
            )
        )

    return tuple(liquid_configs)


def _make_enthalpy_of_vaporization_compute_configs(
    data_entry: DataEntry,
    force_field: str,
    n_molecules: int,
) -> Sequence[BulkLiquid | VacuumGas]:
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


# TODO: Move this into the step that processes multiple configs, maybe SimulationWorkflow.submit_batch
# TODO: When setting up jobs, could have these key a dict that also stores target densities (for when the property is
#       pure liquid but not density, like dielectric constant or enthalpy of mixing). See Issue #103
def get_liquid_deduplication_key(item: BulkLiquid, ignore_keys: list[str] = list()):
    # TODO: Might want to make this item: BaseComputeConfig

    smiles_sorted, x_sorted = [*map(tuple, zip(*sorted(zip(item["smiles"], item["x"]), key=lambda pair: pair[0])))]
    subset = {
        "force_field": item["force_field"],
        "x": x_sorted,
        "smiles": smiles_sorted,
        "temperature": item["temperature"],
        "pressure": item["pressure"],
        "n_molecules": item["n_molecules"],
    }

    return tuple((k, v) for k, v in subset.items() if k not in ignore_keys)
