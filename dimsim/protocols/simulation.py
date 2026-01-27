import pathlib

from ..configs.protocols import SimulationConfig
from .base import Protocol


class Simulation(Protocol):
    """
    This protocol runs a molecular dynamics simulation on the system.

    Much like the Equilibration protocol,
    this protocol uses conditions to determine when the simulation is complete,
    and will run until those conditions are met,
    or until a maximum simulation length is reached.
    """

    config: SimulationConfig

    @classmethod
    def _get_execution_output_names(cls) -> tuple[str, ...]:
        return ("trajectory_path",)

    def _execute(
        self,
        **kwargs
    ) -> pathlib.Path:

        ...
