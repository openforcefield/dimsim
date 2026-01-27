import typing

from .equilibration import EquilibrationConfig


class SimulationConfig(EquilibrationConfig):
    """
    Configuration for the Simulation protocol.
    """

    name: typing.ClassVar[typing.Literal["Simulation"]] = "Simulation"
