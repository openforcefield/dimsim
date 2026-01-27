
import pathlib

import pydantic

from ..base import BaseModel
from ..properties.properties import PhaseName, PropertyName
from .phase import PhaseConfig
from .properties import PropertyConfigType


class WorkflowConfig(BaseModel):
    """
    Overall configuration for workflows,
    potentially including multiple replicates,
    and per-phase and per-target-type protocol configurations.
    """
    n_replicates: int = 1
    protocols: PhaseConfig = PhaseConfig()
    phases: dict[PhaseName, PhaseConfig] = pydantic.Field(
        default_factory=dict,
        description=(
            "Per-phase workflow protocol configurations. "
            "These will override the global protocol configurations defined "
            "in 'protocols'. "
            "The keys are phase names."
        ),
    )
    targets: dict[PropertyName, PropertyConfigType] = pydantic.Field(
        default_factory=dict,
        description=(
            "Per-target and per-phase workflow protocol configurations. "
            "These will override the global protocol configurations defined in 'protocols' and 'phases'. "
            "The structure is a nested dictionary where the first key is the target name "
            "and the second key is the phase name."
        ),
    )

    def get_protocol_config(
        self,
        target_type: PropertyName,
        phase: PhaseName,
    ):
        try:
            return self.targets[target_type][phase]
        except KeyError:
            try:
                return self.phases[phase]
            except KeyError:
                return self.protocols

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "WorkflowConfig":
        """Load a WorkflowConfig from a YAML file.

        Parameters
        ----------
        path : str | pathlib.Path
            Path to the YAML configuration file.

        Returns
        -------
        WorkflowConfig
            The loaded WorkflowConfig instance.
        """
        import yaml

        with open(path, 'r') as f:
            yaml.safe_load(f)

        # TODO: use `protocols` as a default base
        # and patch in phases,
        # then patch in targets

        # so for example, ideally just setting n_max_molecules
        # at the top level changes it for everything

        raise NotImplementedError("Loading from YAML is not yet implemented.")
