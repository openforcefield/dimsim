from ..configs.protocols.coordinates import CoordinateGenerationConfig
from ..coordinates.box import BoxCoordinates
from .base import Protocol


class CoordinateGeneration(Protocol):
    """
    Protocol for querying or generating coordinates for a given substance
    """
    config: CoordinateGenerationConfig

    @classmethod
    def _get_execution_outputs(cls):
        return ("box",)


    def _execute(
        self,
        box: BoxCoordinates,
        **kwargs,
    ) -> BoxCoordinates:
        """
        Execute the CoordinateGeneration protocol.

        Parameters
        ----------
        box : BoxCoordinates
            The initial BoxCoordinates object.
        openmm_system : openmm.System
            The OpenMM System object.

        Returns
        -------
        BoxCoordinates
            The updated BoxCoordinates object with generated coordinates.
        """

        ...
