"""Prep (simulation) jobs from (thermophysical) data entries."""

def compute_configs_from_data_entries(
    data_entries: list[dict],
    force_field: str,
    n_molecules: int,
) -> list[dict]:
    """Convert a list of thermophysical data entries into a list of simulation configs."""
    compute_configs = list()

    for data_entry in data_entries:
        these_configs = _compute_configs_from_data_entry(data_entry, force_field, n_molecules)
        compute_configs.extend(these_configs)

    return compute_configs

def _compute_configs_from_data_entry(
    data_entry: dict,
    force_field: str,
    n_molecules: int,
) -> list[dict]:
    """Convert a single thermophysical data entry into a list of simulation configs."""
    match data_entry:
        case {"tag": "density"}:
            return _make_liquid_density_comput_configs(data_entry, force_field, n_molecules)
        case {"tag": "enthalpy_of_mixing"}:
            return _make_enthalpy_of_mixing_compute_configs(data_entry, force_field, n_molecules)
        case _:
            print(1)
            raise ValueError(f"Unsupported data entry tag: {data_entry['tag']}")

    return [dict()]


def _make_liquid_density_comput_configs(
    data_entry: dict,
    force_field: str,
    n_molecules: int,
) -> list[dict]:
    from dimsim.configs.liquid import BulkLiquid

    return [
        BulkLiquid(
            tag="liquid",
            force_field=force_field,
            n_molecules=n_molecules,
            smiles=data_entry["smiles"],
            x=data_entry["x"],
            temperature=data_entry["temperature"],
            pressure=data_entry["pressure"],
            value=data_entry["value"],
        )
    ]

def _make_enthalpy_of_mixing_compute_configs(
    data_entry: dict,
    force_field: str,
    n_molecules: int,
) -> list[dict]:
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
            value=data_entry["value"],
        )
    ]

    for component_smiles, component_x in zip(data_entry["smiles"], data_entry["x"]):
        liquid_configs.append(
            BulkLiquid(
                tag="liquid",
                force_field=force_field,
                n_molecules=n_molecules,
                smiles=[component_smiles],
                x=[1.0],
                temperature=data_entry["temperature"],
                pressure=data_entry["pressure"],
                value=data_entry["value"],
            )
        )
    
    return liquid_configs