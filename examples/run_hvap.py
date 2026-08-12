from rich import print

from dimsim.compute.configs import local_config, slurm_config
from dimsim.compute.workflow import SimulationWorkflow
from dimsim.datasets.thermoml import ThermoMLDataSet

dataset = ThermoMLDataSet.from_xml(open("dimsim/_tests/data/thermoml/single_dhvap.xml").read())
dhvap_target = dataset.properties[0]

job_specs = list()

base_dir = "dhvap_example"


# production on GPU cluster
if False:
    with SimulationWorkflow(base_dir, slurm_config("gpu")) as workflow:
        pass

# local testing
with SimulationWorkflow(base_dir, local_config(max_workers=10)) as workflow:
    for extra_molecules in range(2):
        workflow.submit_target(
            dhvap_target,
            force_field="openff-2.3.0.offxml",
            n_molecules=500 + extra_molecules,
            n_replicates=2,
        )

    print(f"{workflow._target_compute_mapping=}")

    for extra_molecules in range(2):
        workflow.estimate_target(
            dhvap_target,
            force_field="openff-2.3.0.offxml",
            n_molecules=500 + extra_molecules,
            n_replicates=2,
        )
