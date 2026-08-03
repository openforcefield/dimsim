"""Get observables from production runs."""

import pathlib
from collections import defaultdict

import descent
import openmm.unit
import smee
import smee.converters
import smee.mm
import smee.observables
import tqdm
from descent.targets.thermo import _Observables
from openff.toolkit import ForceField, Molecule, Topology

from dimsim.configs.liquid import BulkLiquid
from dimsim.configs.targets.thermo import DataEntry


def _compute_config_to_descent_entry(
    compute_config: BulkLiquid,
) -> descent.targets.thermo.DataEntry:

    return descent.targets.thermo.DataEntry(
        type="density",
        smiles_a=compute_config["smiles"][0],
        x_a=compute_config["x"][0],
        smiles_b=compute_config["smiles"][1] if len(compute_config["smiles"]) > 1 else None,
        x_b=compute_config["x"][1] if len(compute_config["x"]) > 1 else None,
        temperature=compute_config["temperature"],
        pressure=compute_config["pressure"],
        value=compute_config["density"],
        std=0.0,  # not tracked in compute config
        units="g/cc",  # probably?
        source="",  # also not tracked at level of compute config
    )


def _target_to_descent_entry(target: DataEntry) -> descent.targets.thermo.DataEntry:

    # no tracking of phase in descent's object

    # DataType = typing.Literal["density", "hvap", "hmix"]
    # "type" is DataType in descent
    _tag_type_mapping = {
        "density": "density",
        "enthalpy_of_vaporization": "hvap",
        "enthalpy_of_mixing": "hmix",
    }

    return descent.targets.thermo.DataEntry(
        type=_tag_type_mapping[target["tag"]],
        smiles_a=target["smiles"][0],
        x_a=target["x"][0],
        smiles_b=target["smiles"][1] if len(target["smiles"]) > 1 else None,
        x_b=target["x"][1] if len(target["x"]) > 1 else None,
        temperature=target["temperature"],
        pressure=target["pressure"],
        value=target["value"],
        std=target["std"],
        units=target["units"],
        source=target["source"],
    )


def get_observables(
    compute_config: BulkLiquid,
    job_dir: str,
) -> _Observables:
    # easier if we just serialize the Interchange in each directory, but it sure
    # feels like there should be a better way of handling all of these objects
    topology = Topology.from_pdb(
        f"{job_dir}/production_topology.pdb",
        unique_molecules=[Molecule.from_smiles(smiles) for smiles in compute_config["smiles"]],
    )
    interchanges = [ForceField(compute_config["force_field"]).create_interchange(topology)]

    tensor_force_field, tensor_topologies = smee.converters.convert_interchange(interchanges)

    # not used right now?
    _tensor_system = smee.TensorSystem(topologies=tensor_topologies, n_copies=[1], is_periodic=True)

    # in only one job, this is going to be a single entry, at least for density
    descent_entries = [_compute_config_to_descent_entry(compute_config)]

    required_simulations, entry_to_simulation = descent.targets.thermo._plan_simulations(
        descent_entries, tensor_topologies
    )

    for descent_entry, descent_keys in tqdm.tqdm(
        zip(descent_entries, entry_to_simulation, strict=True),
        desc="Computing observables for each target",
        ncols=80,
        total=len(descent_entries),
    ):
        # Josh has a `per_type_scales` variable floating around
        type_scale = 1.0

        observables: defaultdict[str, descent.targets.thermo.DataEntry] = defaultdict(dict)

        predicted = list()

        for descent_key in descent_keys.values():
            temperature = descent_key["temperature"] * openmm.unit.kelvin
            pressure = None if descent_key["pressure"] is None else descent_key["pressure"] * openmm.unit.atmospheres
            obs = descent.targets.thermo._Observables(
                *smee.mm.compute_ensemble_averages(
                    system=required_simulations["bulk"][descent_key],
                    force_field=tensor_force_field,
                    frames_path=pathlib.Path(f"{job_dir}/production_trajectory.msgpack"),
                    temperature=temperature,
                    pressure=pressure,
                ),
            )
            observables["bulk"][descent_key] = obs

        # could end here? do we need to return the predictions?

        pred, _ = descent.targets.thermo._predict(
            entry=descent_entry,
            keys=descent_keys,
            observables=observables,
            systems=required_simulations,
        )

        predicted.append(pred * type_scale)

    # again unclear what we want to return
    return observables, pred
