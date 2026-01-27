
import openmm

from ..configs.protocols import MinimizationConfig
from ..coordinates.box import BoxCoordinates
from .base import Protocol


class Minimization(Protocol):
    """
    This protocol performs energy minimization on the system.
    """
    config: MinimizationConfig

    @classmethod
    def _get_execution_outputs(cls):
        return ("box", "openmm_state",)

    def _execute(
        self,
        **kwargs
    ) -> tuple[BoxCoordinates, openmm.State]:
        ...
