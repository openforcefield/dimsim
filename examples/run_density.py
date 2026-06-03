from dimsim.compute.configs import local_config, slurm_config
from dimsim.compute.workflow import SimulationWorkflow
from dimsim.configs.compute.density import DensityConfig
from dimsim.datasets.thermoml import ThermoMLDataSet

dataset = ThermoMLDataSet.from_xml(open("dimsim/_tests/data/thermoml/single_density.xml").read())

job_specs = list()

for target in dataset.properties:
    for extra_n_molecules in range(20):
        job_specs.append(
            DensityConfig(
                tag="density",
                target=target,
                force_field="openff-2.3.0.offxml",
                n_molecules=200 + extra_n_molecules,
            )
        )

# local testing
with SimulationWorkflow("outputs", local_config()) as workflow:
    results = workflow.run(job_specs)

# production
if False:
    with SimulationWorkflow("outputs", slurm_config("gpu")) as workflow:
        results = workflow.run(job_specs)
