
import pathlib
import typing

import torch

import descent.targets.thermo
import smee.mm
from ..utils.openmm import SimulationTensorReporter
from ..configs.properties import PropertyConfigType
from smee.mm._ops import (
    _unpack_force_field,
)

class _OpKwargs(typing.TypedDict):
    """The keyword arguments passed to the custom PyTorch op for computing ensemble averages."""

    # force field kwargs
    force_field: smee.TensorForceField
    parameter_lookup: dict[str, int]
    attribute_lookup: dict[str, int]
    has_v_sites: bool

    system: smee.TensorSystem

    frames_path: pathlib.Path

    beta: float
    pressure: float | None
    custom_forcegroups: dict[str, int]
    entry: descent.targets.thermo.DataEntry
    property_config: PropertyConfigType


def _compute_base_observables(
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    frames_file: typing.BinaryIO,
    theta: tuple[torch.Tensor],
    custom_forcegroups: dict[str, int],
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor | None]]:
    """
    Compute base observables: potential energy, kinetic energy, and volume.

    Parameters
    ----------
    system : smee.TensorSystem
        The system to compute observables for.
    force_field : smee.TensorForceField
        The force field to use for energy calculations.
    frames_file : typing.BinaryIO
        A binary file-like object containing the trajectory frames.
    theta : tuple[torch.Tensor]
        Model parameters. This is a flat list of tensors in the following order:
        potential parameter tensors, potential attribute tensors, and vsite parameter tensors.
        Parameter tensors have shape (n_parameters, n_parameter_cols)
        (e.g. 50 parameters with two cols representing sigma and epsilon would have shape (50, 2)).
        Attribute tensors have shape (n_attributes, ) (e.g. (4,) for
        scale12, scale13, scale14, scale15).
        Vsite parameter tensors have shape (n_vsite_parameters, 3).
        The 3 columns represent distance, in-plane angle, and out-of-plane angle.
    """

    needs_grad = [i for i, v in enumerate(theta) if v is not None and v.requires_grad]
    du_d_theta = [None if i not in needs_grad else [] for i in range(len(theta))]
    grad_inputs = [theta[i] for i in needs_grad]

    values = {
        "potential_energy": [],
        "kinetic_energy": [],
        "volume": [],
        "box_vectors": [],
    }
    for key in custom_forcegroups.keys():
        values[key] = []

    reporter = SimulationTensorReporter(
        frames_file,
        # dummy values, this shouldn't matter
        report_interval=1,
        beta=1.0,
        pressure=None,
        custom_forcegroups=custom_forcegroups,
    )
    for framedict in reporter.unpack_frames_as_dict():
        coords = framedict["coordinates"].to(theta[0].device)
        box_vectors = framedict["box_vectors"].to(theta[0].device)
        
        with torch.enable_grad():
            potential = smee.compute_energy(system, force_field, coords, box_vectors)

        if len(grad_inputs):
            du_d_theta_subset: tuple[torch.Tensor] = torch.autograd.grad(
                potential,
                grad_inputs,
                [smee.utils.ones_like(1, potential)],
                retain_graph=False,
                allow_unused=True,
            )

            for idx, i in enumerate(needs_grad):
                du_d_theta[i].append(du_d_theta_subset[idx].float())

        values["box_vectors"].append(box_vectors.detach())
        values["potential_energy"].append(potential.detach())
        values["kinetic_energy"].append(framedict["kinetic_energy"])
        values["volume"].append(torch.det(box_vectors))
        for key in custom_forcegroups.keys():
            values[key].append(framedict[key])
    
    # stack into tensor such that shape is (*, n_frames)
    du_d_theta_array = [
        v if v is None
        else torch.stack(v, dim=-1) for v in du_d_theta
    ]
    return values, du_d_theta_array


class BaseOp(torch.autograd.Function):
    

    @staticmethod
    def _compute_base_observables(
        kwargs: _OpKwargs, *theta: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor | None]]:
        force_field = _unpack_force_field(
            theta,
            kwargs["parameter_lookup"],
            kwargs["attribute_lookup"],
            kwargs["has_v_sites"],
            kwargs["force_field"],
        )
        system = kwargs["system"]
        custom_forcegroups = kwargs["custom_forcegroups"]

        with kwargs["frames_path"].open("rb") as file:
            values, du_d_theta = _compute_base_observables(
                system, force_field, file, theta, custom_forcegroups
            )

        return values, du_d_theta
    


