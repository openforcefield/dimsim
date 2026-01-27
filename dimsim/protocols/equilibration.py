
import openmm

from ..configs.protocols import EquilibrationConfig, InitialEquilibrationConfig
from ..coordinates.box import BoxCoordinates
from .base import Protocol


class InitialEquilibration(Protocol):
    """
    This protocol conducts an initial equilibration of the system.
    This is less structured than the Equilibration protocol,
    and only accepts a set simulation length.
    It is intended for 2-stage equilibrations,
    e.g. NVT + NPT, or NPT + NVT.
    """

    config: InitialEquilibrationConfig

    @classmethod
    def _get_execution_outputs(cls):
        return ("openmm_state",)

    def _execute(
        self,
        **kwargs
    ) -> openmm.State:
        raise NotImplementedError(
            "InitialEquilibration protocol is not yet implemented."
        )


class Equilibration(Protocol):
    """
    This protocol conducts the main equilibration of the system.
    It uses conditions to determine when the system is equilibrated,
    and will run until those conditions are met,
    or until a maximum simulation length is reached.
    """

    config: EquilibrationConfig

    @classmethod
    def _get_execution_outputs(cls):
        return ("box", "openmm_state",)

    def _execute(
        self,
        **kwargs
    ) -> tuple[BoxCoordinates, openmm.State]:
        raise NotImplementedError("Equilibration protocol is not yet implemented.")
