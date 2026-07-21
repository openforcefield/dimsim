from dimsim.compute.configs import local_config, slurm_config
from dimsim.compute.fetch import fetch_trajectory_paths_from_target
from dimsim.compute.workflow import SimulationWorkflow
from dimsim.datasets.thermoml import ThermoMLDataSet

dataset = ThermoMLDataSet.from_xml(open("dimsim/_tests/data/thermoml/single_density.xml").read())
density_target = dataset.properties[0]

job_specs = list()

base_dir = "jobs"


# production on GPU cluster
if False:
    with SimulationWorkflow(base_dir, slurm_config("gpu")) as workflow:
        pass

# local testing
with SimulationWorkflow(base_dir, local_config()) as workflow:
    for extra_molecules in range(2):
        workflow.submit_target(
            density_target,
            force_field="openff-2.3.0.offxml",
            n_molecules=200 + extra_molecules,
        )

# TODO: Show how to check status while running

# get trajectory paths from root job directory and per-target info (without knowing internal compute configs)
trajectory_paths = [
    fetch_trajectory_paths_from_target(
        base_dir=base_dir,
        target=density_target,
        force_field="openff-2.3.0.offxml",
        n_molecules=200 + extra_molecules,
    )
    for extra_molecules in range(2)
]
print(trajectory_paths)
# [('jobs/f88a3e081bc694c0e4bec5d25332f116d7543de667a5e1de0dca3455795be2fb/production_trajectory.dcd',),
#  ('jobs/083d52acc879707f21ca1531c3a7285dda9752b263e6966f42de276743ec8f89/production_trajectory.dcd',)]
