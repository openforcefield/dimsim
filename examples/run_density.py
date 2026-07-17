from dimsim.compute.configs import local_config, slurm_config
from dimsim.compute.workflow import SimulationWorkflow
from dimsim.datasets.thermoml import ThermoMLDataSet

dataset = ThermoMLDataSet.from_xml(open("dimsim/_tests/data/thermoml/single_density.xml").read())
density_target = dataset.properties[0]

job_specs = list()

# local testing
with SimulationWorkflow("jobs", local_config()) as workflow:
    for extra_molecules in range(2):
        workflow.submit_target(
            density_target,
            force_field="openff-2.3.0.offxml",
            n_molecules=200 + extra_molecules,
        )

# TODO: Show how to get results by only knowing targets (+force field + n_molecules) and no internal IDs
# TODO: Show how to check status while running

# production
if False:
    with SimulationWorkflow("outputs", slurm_config("gpu")) as workflow:
        pass
