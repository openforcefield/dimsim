import copy
import functools
import warnings
from collections import defaultdict

import datasets
import descent.optim
import descent.utils
import smee.ff
import torch
from distributed import Future

from ..configs.workflow import PropertyConfigType, WorkflowConfig
from ..coordinates.box import BoxCoordinates
from ..properties.properties import NAMES_TO_PROPERTY_TYPES
from ..server.client import Client
from .task import BoxKey, _Task


def simulation_key_to_tensor_system(
    simulation_key: descent.targets.thermo.SimulationKey,
    topologies: dict[str, smee.TensorTopology],
) -> smee.TensorSystem:
    """Convert a SimulationKey to a TensorSystem."""
    system_topologies = []
    n_copies = []
    for smiles, n_mols in zip(simulation_key.smiles, simulation_key.n_molecules):
        if smiles not in topologies:
            raise ValueError(
                f"Topology for SMILES '{smiles}' not found in provided "
                f"topologies."
            )
        system_topologies.append(topologies[smiles])
        n_copies.append(n_mols)
    return smee.TensorSystem(system_topologies, n_copies, True)


# TODO: this is probably a terrible name
class EntrySimulations:
    """
    Accounting object to track which simulations need to be run for each entry,
    as well as the boxkey for each.

    Attributes
    ----------
    entry : dict
        The dataset entry this object corresponds to.
        This should be the descent.targets.thermoml.DataEntry format.
    pure_a : list[BoxKey]
        The box key/s for pure component A simulations, if any.
        This is a list to accommodate multiple replicates.
    pure_b : list[BoxKey]
        The box key/s for pure component B simulations, if any.
        This is a list to accommodate multiple replicates.
    bulk : list[BoxKey]
        The box key/s for bulk mixture simulations, if any.
        This is a list to accommodate multiple replicates.
    vacuum : list[BoxKey]
        The box key/s for vacuum simulations, if any.
        This is a list to accommodate multiple replicates.
    property_type : PropertyType
        The property type of this entry, derived from the entry data.
    """

    def __init__(self, entry):
        self.entry = entry
        self.pure_a = []
        self.pure_b = []
        self.bulk = []
        self.vacuum = []

        self.pure_a_boxkey = None
        self.pure_b_boxkey = None
        self.bulk_boxkey = None
        self.vacuum_boxkey = None

    @property
    def property_type(self):
        data_type = self.entry["type"].lower()
        return NAMES_TO_PROPERTY_TYPES.get(data_type)

    def get_boxkeys(self) -> dict[str, BoxKey]:
        """Get all box keys for this entry as a dict with phase names as keys."""
        boxkeys = {}
        if self.bulk_boxkey:
            boxkeys["bulk"] = self.bulk_boxkey
        if self.pure_a_boxkey:
            boxkeys["pure_a"] = self.pure_a_boxkey
        if self.pure_b_boxkey:
            boxkeys["pure_b"] = self.pure_b_boxkey
        if self.vacuum_boxkey:
            boxkeys["vacuum"] = self.vacuum_boxkey
        return boxkeys


class DimsimWorkflow:
    """
    Workflow for actually running simulations and obtaining predictions.

    This is mostly a convenience class to bundle together
    the various protocols and steps needed to go from a dataset entry
    to simulation predictions.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset containing entries to simulate.
        This should follow the descent.targets.thermoml.DataEntry format.
    device : torch.device | str, optional
        The device to run simulations on, by default "cpu".
        This workflow does not really use the pytorch device directly,
        but it is included for consistency with other components.
    """

    def __init__(
        self,
        dataset: datasets.Dataset,
        trainable,
        topologies: dict[str, smee.TensorTopology],
        config: WorkflowConfig = WorkflowConfig(),
    ):

        self.entries = tuple(*descent.utils.dataset.iter_dataset(dataset))
        self.config = config
        self.trainable = trainable
        self.topologies = topologies

        # we do this in __init__ so we can inspect and modify as needed
        self._entry_simulations, self._hash_to_config = self.generate_entry_simulations()

    def generate_entry_simulations(
        self,
    ) -> tuple[list[EntrySimulations], dict[int, PropertyConfigType]]:
        """
        Generate the EntrySimulations objects for each dataset entry,
        as well as the unique configuration hashes.

        This determines which simulations need to be run for each entry,
        and the corresponding box keys and configuration hashes.
        Instead of passing the config for each task, we will pass a hash of the config,
        and distribute the unique configs separately to avoid redundant serialization.

        Returns
        -------
        tuple[list[EntrySimulations], dict[int, PropertyConfigType]]
            A tuple containing:
            - A list of EntrySimulations objects for each dataset entry.
            - A dictionary mapping configuration hashes to PropertyConfigType objects.
        """

        entry_simulations: list[EntrySimulations] = []
        hash_to_config: dict[int, PropertyConfigType] = {}

        for entry in self.entries:
            # create and add
            entry_simulation = EntrySimulations(entry)
            entry_simulations.append(entry_simulation)

            property_type = entry_simulation.property_type

            # TODO: tidy this up
            if property_type.requires_bulk_sim:
                # get user-modified bulk config
                config = self.config.get_protocol_config(
                    target_type=property_type.name,
                    phase="bulk",
                )
                config_hash = config._get_hash()
                hash_to_config[config_hash] = config

                # generate box with correct n_molecules
                n_max_mols = config.coordinate_generation.n_max_molecules
                box = BoxCoordinates.from_data_entry(
                    entry,
                    n_max_mols=n_max_mols,
                )
                boxkey = BoxKey(
                    simulation_key=box.to_simulation_key(),
                    config_hash=config_hash,
                )

                # and assign to entrysimulation
                entry_simulation.bulk_boxkey = boxkey

            if property_type.requires_pure_sim:
                # user-modified pure config
                config = self.config.get_protocol_config(
                    target_type=property_type.name,
                    phase="pure",
                )
                config_hash = config._get_hash()
                hash_to_config[config_hash] = config
                n_max_mols = config.coordinate_generation.n_max_molecules

                for smiles_key in ["smiles_a", "smiles_b"]:
                    # directly make a simulationkey here
                    simulation_key = descent.targets.thermo.SimulationKey(
                        (entry[smiles_key],),
                        (n_max_mols,),  # all molecules of this type
                        entry["temperature"],
                        entry["pressure"],
                    )
                    boxkey = BoxKey(
                        simulation_key=simulation_key, config_hash=config_hash,
                    )
                    if smiles_key == "smiles_a":
                        entry_simulation.pure_a_boxkey = boxkey
                    else:
                        entry_simulation.pure_b_boxkey = boxkey

            if property_type.requires_vacuum_sim:
                # need to check a bunch of things here
                assert (
                    entry["smiles_b"] is None
                ), "Vacuum simulations only supported for single molecules."

                # we copy and modify the config as needed
                config = self.config.get_protocol_config(
                    target_type=property_type.name,
                    phase="vacuum",
                ).model_copy(deep=True)

                n_max_mols = config.coordinate_generation.n_max_molecules
                if n_max_mols != 1:
                    # TODO: only warn once?
                    warnings.warn(
                        "n_max_molecules is ignored for vacuum simulations; "
                        "only one molecule will be simulated."
                    )
                    config.coordinate_generation.n_max_molecules = 1

                stages = [
                    "initial_equilibration",
                    "equilibration",
                    "simulation",
                ]
                for stage in stages:
                    stage_config = getattr(config, stage)
                    if stage_config.ensemble == "NPT":
                        warnings.warn(
                            f"NPT ensemble is not applicable for vacuum simulations in {stage} "
                            "and will be changed to NVT."
                        )
                        setattr(stage_config, "ensemble", "NVT")

                config_hash = config._get_hash()
                hash_to_config[config_hash] = config

                simulation_key = descent.targets.thermo.SimulationKey(
                    (entry["smiles_a"],),
                    (1,),  # single molecule in vacuum
                    entry["temperature"],
                    None,
                )
                boxkey = BoxKey(
                    simulation_key=simulation_key,
                    config_hash=config_hash,
                )
                entry_simulation.vacuum_boxkey = boxkey

        return entry_simulations, hash_to_config

    def run_simulations(self, force_field: smee.ff.TensorForceField):
        """
        Run the simulations in a non-distributed manner on the current machine.
        """

        # TODO: maybe do this better instead of hackily
        from distributed import Client, LocalCluster

        with LocalCluster(n_workers=1, threads_per_worker=1) as cluster:
            with Client(cluster) as client:
                return self.run_distributed_simulations(client, force_field)

    def run_distributed_simulations(
        self, client, force_field: smee.ff.TensorForceField
    ):
        """
        Run the simulations in a distributed manner using the provided Client.
        """

        # distribute configs in memory
        # we want to be able to access these on all workers
        # and just seralize the hash with each task
        hashed_configs_future = client.scatter(self._hash_to_config, broadcast=True)

        # get unique boxes so we deduplicate simulation tasks
        # main outcome is to avoid simulating a million water boxes
        unique_box_keys: set[BoxKey] = set()
        for entry in self._entry_simulations:
            unique_box_keys |= set(entry.get_boxkeys().values())

        # first create coordinate generation tasks
        # this doesn't need multiple replicates
        # here is where we either grab an existing or pack a new box
        coordinate_tasks = []
        for box_key in unique_box_keys:
            box_coordinates = BoxCoordinates.from_simulation_key(box_key.simulation_key)
            task = _Task(
                box_key,
                config_names=["coordinate_generation"],
                inputs={"box": box_coordinates},
            )
            coordinate_tasks.append(task)

        # submit coordinate generation tasks
        futures = client.submit(coordinate_tasks, hashed_configs_future)

        # gather coordinate generation outputs
        output_tasks = []
        for chunk in client.as_completed(futures, chunksize=10):
            # as they complete, check if we need to store coordinates
            # do this in chunks to avoid too many small operations,
            # but only store in the main process to avoid race conditions
            for task in chunk:
                config = self._hash_to_config[task.config_hash]
                if config.coordinate_generation.store_packed_coordinates:
                    # store coordinates
                    raise NotImplementedError("Coordinate storage not implemented yet.")
                output_tasks.append(task)

        # ===== minimization and equilibration tasks =====
        # here we need replicates
        equilibration_tasks_base = []
        for task in output_tasks:
            # do one replicate first to ensure no immediate crashes
            inputs = dict(task.outputs)

            # at some point we need to convert to the smee type necessary
            # could happen either in the main process or in the worker
            # not sure which is better yet
            smee_system = simulation_key_to_tensor_system(
                task.box_key.simulation_key, self.topologies
            )
            inputs["smee_system"] = smee_system
            inputs["smee_force_field"] = force_field
            equil_task = _Task(
                config_hash=task.config_hash,
                # run the system generation, minimization, and equilibration
                # all at once -- no need to break down too much further here
                config_names=[
                    "system_generation",
                    "minimization",
                    "initial_equilibration",
                    "equilibration",
                ],
                inputs=inputs,
                replicate=1
            )
            equilibration_tasks_base.append(equil_task)

        # add additional replicates by copying and modifying replicate number
        additional_equilibration_tasks = []
        if self.config.n_replicates > 1:
            for rep in range(2, self.config.n_replicates + 1):
                # copying keep all the same input, and crucially the uuid
                _task_list = [copy.copy(x) for x in equilibration_tasks_base]
                for task in _task_list:
                    task.replicate = rep
                additional_equilibration_tasks.extend(_task_list)

        # submit all tasks at once
        equilibration_tasks = equilibration_tasks_base + additional_equilibration_tasks
        equilibration_futures = client.submit(
            equilibration_tasks, hashed_configs_future
        )

        # monitor equilibration completion and submit simulation tasks as they come in
        simulation_futures: list[Future] = []
        # we also need to track replicates for storing equilibrated coordinates
        equilibration_replicates = defaultdict(lambda: defaultdict(list))

        # again, best chunksize is just a guess here
        for chunk in client.as_completed(equilibration_futures, chunksize=20):
            for task in chunk:
                # track replicates
                uuid_ = task._task_id
                equilibration_replicates[task.config_hash][uuid_].append(task)

                # submit simulation task for this completed equilibration
                simulation_task = _Task(
                    task.box_key,
                    config_names=["simulation"],
                    inputs=dict(task.outputs),
                    replicate=task.replicate,
                )
                simulation_futures.append(
                    client.submit(simulation_task, hashed_configs_future)
                )

            # check for completed equilibrations to add to store
            finished = defaultdict(list)
            for config_hash, replicate_dict in equilibration_replicates.items():
                # no need for this if we aren't storing
                config = self.config._hash_to_config[config_hash]
                if not config.equilibration.store_equilibrated_coordinates:
                    continue

                # do we only store the lowest of the three replicates?
                only_lowest = config.equilibration.store_lowest_energy_replicate_only
                for uuid_, tasks in replicate_dict.items():
                    # skip if not all replicates are done yet
                    if len(tasks) < self.config.n_replicates:
                        continue
                    if len(tasks) > self.config.n_replicates:
                        raise ValueError(
                            f"More replicate tasks than expected for config {config_hash}. "
                            f"Expected {self.config.n_replicates}, got {len(tasks)}."
                        )

                    to_store = [task.outputs["box"] for task in tasks]
                    if only_lowest:
                        to_store = [BoxCoordinates.get_lowest_energy_box(to_store)]
                    for box in to_store:
                        # store coordinates
                        raise NotImplementedError(
                            "Coordinate storage not implemented yet."
                        )

                    finished[config_hash].append(uuid_)

            # remove finished replicates from the dict to avoid rechecking
            for config_hash, uuids in finished.items():
                for uuid_ in uuids:
                    equilibration_replicates[config_hash].pop(uuid_)


        # get output simulation tasks
        output_simulation_tasks: dict[BoxKey, list[_Task]] = defaultdict(list)
        for chunk in client.as_completed(simulation_futures, chunksize=20):
            for task in chunk:
                box_key = simulation_futures[task]
                output_simulation_tasks[box_key].append(task)

        # or store the paths for each box_key at the trajectory storage?
        # TODO: not sure what's best to return here
        return output_simulation_tasks



    def compute_properties_gathered(
        self,
        force_field: smee.ff.TensorForceField,
        simulation_outputs: dict[BoxKey, list[_Task]],
    ) -> list[tuple[descent.targets.thermoml.DataEntry, float]]:
        """
        Compute properties based on the gathered simulation outputs, on a single machine.

        Parameters
        ----------
        force_field : smee.ff.TensorForceField
            The force field used for the simulations.
        simulation_outputs : dict[BoxKey, list[_Task]]
            A dictionary mapping BoxKeys to lists of completed simulation Tasks.
            This is the output of ``run_distributed_simulations``,
            unless the trajectory storage database has user-friendly
            methods defined to access simulation outputs directly.
        """

        results = []

        for entry_simulation in self._entry_simulations:
            property_type = entry_simulation.property_type
            if property_type is None:
                raise ValueError(
                    f"Unknown property type '{entry_simulation.entry['type']}' "
                    "for dataset entry."
                )

            # gather necessary simulation outputs
            box_keys = entry_simulation.get_boxkeys()
            simulation_kwargs = {}
            for phase, boxkey in box_keys.items():
                if boxkey not in simulation_outputs:
                    raise ValueError(
                        f"Missing simulation outputs for phase '{phase}' "
                        f"and box key '{boxkey}'."
                    )
                paths = [
                    task.outputs["trajectory_path"] for task in simulation_outputs[boxkey]
                ]
                simulation_kwargs[f"{phase}_trajectory_paths"] = paths

            # calculate property
            property_value = property_type.calculate_property(
                entry_simulation.entry,
                **simulation_kwargs
            )
            results.append((entry_simulation.entry, property_value))

        return results

    def get_closure_fn(
        self,
        client: Client | None = None,
        # TODO: allow passing in custom loss function
    ) -> descent.optim.ClosureFn:
        """Get a closure function for optimization."""

        if client is None:
            run_function = self.run_simulations
        else:
            run_function = functools.partial(self.run_distributed_simulations, client)

        def closure_fn(
            x: torch.Tensor,
            compute_gradient: bool,
            compute_hessian: bool,
        ):
            gradient = None
            hessian = None

            # current FF -- detach for passing to workers
            detached_ff: smee.ff.TensorForceField = self.trainable.to_force_field(
                x.detach().clone()
            )
            output_simulation_tasks = run_function(detached_ff)

            # TODO: add ability to also compute properties and gradients in a distributed manner
            # recreate ff on main process to make sure autograd works locally
            current_ff = self.trainable.to_force_field(x)
            property_results = self.compute_properties_gathered(
                current_ff,
                output_simulation_tasks
            )
            # TODO: add ability to scale predicted property, perhaps within the property config
            y_pred = torch.tensor(
                [result[1] for result in property_results],
                dtype=x.dtype,
                device=x.device,
            )
            y_ref = torch.tensor(
                [
                    entry["value"] for entry in self.entries
                ],
                dtype=x.dtype,
                device=x.device,
            )

            # TODO: allow this to be custom
            loss = ((y_pred - y_ref) ** 2).sum()

            if compute_gradient:
                gradient = torch.autograd.grad(loss, x, retain_graph=True)[0].detach()

            if compute_hessian:
                # assumes sum of squares loss
                hessian = descent.utils.loss.approximate_hessian(x, y_pred).detach()

            return loss.detach(), gradient, hessian

        return closure_fn
