import typing
import uuid

import descent

from .config import WorkflowConfig


class BoxKey(typing.NamedTuple):
    simulation_key: descent.targets.thermo.SimulationKey
    config_hash: int

class _Task:
    """
    A Task represents a single workflow execution, which may consist of multiple protocols.
    It contains a configuration hash, a list of protocol names to execute,
    and the inputs/outputs for the first protocol.
    It contains the configuration for the workflow and the inputs/outputs for each protocol.

    # TODO: this is mostly a temporary convenience class
    # and maybe shouldn't live here.
    """

    def __init__(
        self,
        box_key: BoxKey,
        config_names: list[str],
        inputs: dict,
        replicate: int = 1
    ):
        self.box_key = box_key
        self.config_names = config_names
        self.inputs = inputs
        self.outputs = dict(inputs)
        self.replicate = replicate
        self._task_id = uuid.uuid4()

    @property
    def config_hash(self) -> int:
        return self.box_key.config_hash

    @property
    def simulation_key(self) -> descent.targets.thermo.SimulationKey:
        return self.box_key.simulation_key

    def execute(self, hash_to_config: dict[int, WorkflowConfig]):
        config = hash_to_config[self.config_hash]
        self.execute_from_config(config)

    def execute_from_config(self, config: WorkflowConfig):
        for protocol_names in self.config_names:
            protocol_configs = getattr(config, protocol_names, [])
            if not isinstance(protocol_configs, list):
                protocol_configs = [protocol_configs]

            for protocol_config in protocol_configs:
                protocol_class = protocol_config.get_protocol_class()
                protocol = protocol_class(config=protocol_config)
                protocol.execute_on_task(self)

        return self
