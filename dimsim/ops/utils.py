import smee
import torch
from openff.units import unit
from openff.units.elements import MASSES as _OPENFF_MASSES


def _compute_mass(system: smee.TensorSystem) -> float:
    """Compute the total mass of a system."""

    def _get_mass(v: int) -> float:
        return _OPENFF_MASSES[int(v)].m_as(unit.dalton)

    return sum(
        sum(_get_mass(atomic_num) for atomic_num in topology.atomic_nums) * n_copies
        for topology, n_copies in zip(system.topologies, system.n_copies, strict=True)
    )

def _reshape_du_d_theta(
    du_d_theta: list[torch.Tensor | None]
) -> list[torch.Tensor | None]:
    """Reshape du_d_theta to ensure 3 dimensions for backward pass."""
    return [
        None if v is None
        else (
            v if v.ndim == 3
            else v.unsqueeze(0)
        ) for v in du_d_theta
    ]

def _unpack_context_and_outputs(ctx, grad_outputs):
    """Unpack saved tensors from context."""
    n_theta = ctx.n_theta
    du_d_theta = _reshape_du_d_theta(
        ctx.saved_tensors[n_theta:2 * n_theta]
    )
    stacked_values = ctx.saved_tensors[-2]
    avg_values = ctx.saved_tensors[-1]

    # only take the gradients for the observable values
    n_observable = (len(grad_outputs) - 1) // 2
    grad_outputs = torch.stack(grad_outputs[:n_observable])
    return n_theta, du_d_theta, stacked_values, avg_values, grad_outputs


def _pack_context_and_outputs(
    ctx,
    values: dict[str, torch.Tensor],
    theta: tuple[torch.Tensor],
    du_d_theta: list[torch.Tensor | None],
) -> tuple[torch.Tensor, ...]:
    """
    Pack tensors into context for backward pass.


    Parameters
    ----------
    ctx : torch.autograd.FunctionCtx
        The context to save tensors for backward pass.
    values : dict[str, torch.Tensor]
        The computed observable values.
    theta : tuple[torch.Tensor]
        The model parameters.
    du_d_theta : list[torch.Tensor | None]
        The derivatives of the potential energy with respect to model parameters.

    Returns
    -------
    tuple[torch.Tensor, ...]
        The average observable values, their standard deviations, and column names.
        This tuple has length 2 N + 1, where N is the number of observables.
        The first N entries are the average values, the next N entries are the standard
        deviations, and the last entry is a list of column names.
    """
    # return a flat list of tensors and the column names
    # as pytorch doesn't currently work with arbitrary pytrees
    # see https://github.com/pytorch/pytorch/issues/96337

    column_names = sorted(values.keys())
    stacked_values = torch.stack([values[name] for name in column_names], dim=0)
    stacked_values = stacked_values.to(theta[0].device)
    avg_values = stacked_values.mean(dim=0)
    avg_stds = [*stacked_values.std(dim=0)]

    # modify context for backward pass
    ctx.n_theta = len(theta)
    ctx.columns = column_names
    ctx.save_for_backward(
        *theta,
        *du_d_theta,
        stacked_values,
        avg_values
    )
    ctx.mark_non_differentiable(*avg_stds)

    return *avg_values, *avg_stds, column_names
