"""
Configuration classes for system generation protocols.
"""

import typing

import pydantic

from .base import ProtocolConfig


class GenerateSystemConfig(ProtocolConfig):
    """
    Configuration for the GenerateSystem protocol.
    """

    name: typing.ClassVar[str] = "GenerateSystem"
    additional_forces: dict[str, "AddForceConfig"] = pydantic.Field(
        default_factory=dict,
        description=(
            "A dictionary of additional forces to add to the system. "
            "The keys are the names of the forces, and the values are the corresponding AddForceConfig objects."
        ),
    )


class AddForceConfig(ProtocolConfig):
    """
    Configuration for the AddForce protocol.
    For now, per-item parameters are not supported.

    Examples
    --------

    For a flat-bottom restraint on sodium and chloride ions:

    .. code-block:: python

        config = AddForceConfig(
            expression="0.5*k*max(0, abs(z-z0)-rfb)^2",
            global_parameters={
                "k": "4184 kJ/(nm**2 mol)",
                "rfb": "2.4 nm",
                "z0": "5 nm",
            },
            smarts_patterns=["[#17X0:1]", "[#11X0:1]"], # sodium and chloride ions
            force_type="CustomExternalForce",
        )

    """

    name: typing.ClassVar[str] = "AddForce"
    expression: str = pydantic.Field(
        ...,
        description="The mathematical expression defining the custom force to be added.",
    )
    force_type: typing.Literal["CustomExternalForce"] = pydantic.Field(
        ...,
        description="The type of OpenMM custom force to create.",
    )
    force_name: str = pydantic.Field(
        default="CustomForce",
        description=(
            "The name to assign to the custom force in the OpenMM System. "
            "If you have multiple custom forces, ensure each has a unique name."
        )
    )
    global_parameters: dict[str, str] = pydantic.Field(
        default_factory=dict,
        description=(
            "A dictionary of parameter names and their default values, as strings, for the custom force. "
            "Values with units should be specified and compatible with OpenFF units. "
            "These will be added as global parameters to the force."
        ),
    )
    smarts_patterns: list[str] = pydantic.Field(
        default_factory=list,
        description=(
            "A list of SMARTS patterns corresponding to the per-item parameters. "
            "These will be used to identify which particles the per-item parameters apply to. "
        ),
    )

    whole_molecule: bool = pydantic.Field(
        default=False,
        description=(
            "Whether to apply the force to whole molecules matching the SMARTS patterns. "
            "If True, the force will be applied to all particles in the molecule if any particle matches the SMARTS pattern. "
            "If False, the force will only be applied to the particles that match the SMARTS pattern."
        ),
    )


    @pydantic.field_validator("global_parameters")
    def _validate_parameters(
        cls, v: dict[str, str]
    ) -> dict[str, str]:
        """
        Validate that parameter values include valid OpenMM units.

        Parameters
        ----------
        v : dict[str, str]
            The parameters to validate.

        Returns
        -------
        dict[str, str]
            The validated parameters.

        Raises
        ------
        ValueError
            If any parameter value does not include units.
        """
        ...
