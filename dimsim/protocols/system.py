"""
Protocols to do with converting Smee objects to OpenMM Systems.
"""

import openmm
import smee
from openff.units import unit
from openff.units.openmm import to_openmm

from ..configs.protocols import AddForceConfig, GenerateSystemConfig
from ..coordinates.box import BoxCoordinates
from .base import Protocol

FORCE_NAME_TO_CLASS = {
    "CustomExternalForce": openmm.CustomExternalForce,
}

FORCE_ADDER_FUNCTIONS = {
    openmm.CustomExternalForce: "addParticle",
}


class GenerateSystem(Protocol):
    """
    Protocol to generate an OpenMM System from a Smee TensorSystem and TensorForceField.
    """

    config: GenerateSystemConfig

    @classmethod
    def _get_execution_outputs(cls):
        return ("openmm_system", "custom_force_groups")

    def _execute(
        self,
        box: BoxCoordinates = None,
        smee_system: smee.TensorSystem | None = None,
        smee_force_field: smee.TensorForceField | None = None,
        **kwargs
    ) -> tuple[openmm.System, dict[str, int]]:
        """
        Generate an OpenMM System from the given Smee objects.

        Parameters
        ----------
        _ : BoxCoordinates
            Unused
        smee_system : smee.TensorSystem
            The Smee TensorSystem object.
        smee_force_field : smee.TensorForceField
            The Smee TensorForceField object.

        Returns
        -------
        tuple[openmm.System, dict[str, int]]
            The generated OpenMM System and a dictionary of custom force groups.
            The dictionary keys are the names of the forces,
            and the values are the corresponding force groups assigned to those forces in the OpenMM System.
        """

        assert isinstance(smee_system, smee.TensorSystem), (
            "smee_system must be provided and be a TensorSystem"
        )
        assert isinstance(smee_force_field, smee.TensorForceField), (
            "smee_force_field must be provided and be a TensorForceField"
        )

        openmm_system: openmm.System = ...

        custom_force_groups = {}
        for force_name, force_config in self.config.additional_forces.items():
            add_force_protocol = AddForce(config=force_config)
            openmm_system, force_group = add_force_protocol._execute(
                box=box,
                openmm_system=openmm_system,
            )
            custom_force_groups[force_name] = force_group

        return openmm_system, custom_force_groups


class AddForce(Protocol):
    """
    Protocol to add a custom force to an OpenMM System.
    """

    config: AddForceConfig

    @classmethod
    def _get_execution_outputs(cls):
        return ("force_group",)

    def _execute(
        self,
        box: BoxCoordinates,
        openmm_system: openmm.System,
        **kwargs,
    ) -> int:
        """
        Add the custom force to the OpenMM System.
        A copy is returned.

        Parameters
        ----------
        box : BoxCoordinates
            The BoxCoordinates object.
        openmm_system : openmm.System
            The OpenMM System object to which the force will be added.

        Returns
        -------
        int
            The force group assigned to the added force.
        """

        existing_force_groups = [
            force.getForceGroup() for force in openmm_system.getForces()
        ]
        new_force_group = max(existing_force_groups) + 1 if existing_force_groups else 0

        # add force
        force_type = FORCE_NAME_TO_CLASS[self.config.force_type]
        custom_force = force_type(self.config.expression)
        openmm_system.addForce(custom_force)
        custom_force.setForceGroup(new_force_group)

        # add global parameters
        for param_name, param_value in self.config.global_parameters.items():
            param_quantity = to_openmm(unit.Quantity(param_value))
            custom_force.addGlobalParameter(param_name, param_quantity)


        # identify particles by SMARTS and set per-item parameters
        topology = box.to_openff_topology()
        adder_func_name = FORCE_ADDER_FUNCTIONS[force_type]
        adder = getattr(custom_force, adder_func_name)
        if "Particle" in adder_func_name:
            n_expected = 1
        else:
            raise NotImplementedError(
                f"Adder function '{adder_func_name}' not supported yet."
            )

        # For each SMARTS pattern, find matching particles and add them to the force
        to_add = []
        for smarts in self.config.smarts_patterns:
            matches = topology.chemical_environment_matches(smarts)
            for match in matches:
                if not len(match) == n_expected:
                    raise ValueError(
                        f"Expected {n_expected} matches for SMARTS "
                        f"'{smarts}', got {len(match)}"
                    )
                if self.config.whole_molecule:
                    molecule = topology.atom(match[0].index).molecule
                    for atom in molecule.atoms:
                        to_add.append(atom.index)

        to_add = sorted(set(to_add))
        if "Particle" in adder_func_name:
            for particle_index in to_add:
                adder(particle_index)

        return new_force_group
