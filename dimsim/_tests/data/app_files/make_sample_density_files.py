import json
import pathlib
import random

from dimsim.compute._minimize import _minimize_energy
from dimsim.compute._pack import _prepare_packed_topology
from dimsim.compute._prepare import _prepare_openmm_system
from dimsim.compute.prep import _make_liquid_density_compute_configs
from dimsim.configs.targets.thermo import DataEntry

pathlib.Path("sample_density/").mkdir(exist_ok=True)

target = DataEntry(
    **{
        "id": random.randint(10**15, 10**16 - 1),
        "tag": "density",
        "x": [0.5, 0.5],
        "smiles": [
            "[O:1]([H:2])[H:3]",
            "[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]",
        ],
        "temperature": 298.15,
        "pressure": 1.01325,
        "value": 0.922,
        "std": 0.0001,
        "units": "g/mL",
        "source": "",
    }
)

with open("sample_density/target_config.json", "w") as f:
    json.dump(target, f)

compute = _make_liquid_density_compute_configs(
    data_entry=target,
    force_field="openff-2.3.0.offxml",
    n_molecules=200,
)[0]

packing_result = _prepare_packed_topology(
    compute_config=compute,
    job_dir="sample_density/",
)

prepare_result = _prepare_openmm_system(
    packing_future=packing_result,
    job_dir="sample_density/",
)

minimize_result = _minimize_energy(
    system_future=prepare_result,
    job_dir="sample_density/",
)
