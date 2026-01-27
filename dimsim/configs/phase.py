
import pydantic

from ..base import BaseModel
from .protocols import (
    CoordinateGenerationConfig,
    EquilibrationConfig,
    GenerateSystemConfig,
    InitialEquilibrationConfig,
    MinimizationConfig,
    SimulationConfig,
)


class PhaseConfig(BaseModel):
    """
    Configuration for a single workflow aimed at generating a single simulation
    for a given property and phase.

    #TODO: the name of this class could be improved
    """
    coordinate_generation: CoordinateGenerationConfig = pydantic.Field(
        default_factory=lambda: CoordinateGenerationConfig(),
        description="Coordinate generation protocol to apply.",
    )
    system_generation: GenerateSystemConfig = pydantic.Field(
        default_factory=lambda: GenerateSystemConfig(),
        description="System generation protocol to apply.",
    )
    minimization: MinimizationConfig = pydantic.Field(
        default_factory=lambda: MinimizationConfig(),
        description="Minimization protocol to apply.",
    )
    initial_equilibration: InitialEquilibrationConfig = pydantic.Field(
        default_factory=lambda: InitialEquilibrationConfig(),
        description="Initial equilibration protocol to apply.",
    )
    equilibration: EquilibrationConfig = pydantic.Field(
        default_factory=lambda: EquilibrationConfig(),
        description="Equilibration protocol to apply.",
    )
    simulation: SimulationConfig = pydantic.Field(
        default_factory=lambda: SimulationConfig(),
        description="Production simulation protocol to apply.",
    )
