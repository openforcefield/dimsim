import torch
from openff.units import unit

from .base import BaseOp, _EnsembleAverageKwargs
from .utils import _compute_mass

_DENSITY_CONVERSION = 1.0e24 / unit.avogadro_constant.m_as(
    1 / unit.mole
)

class DensityOp(BaseOp):

    @staticmethod
    def forward(ctx, kwargs: _EnsembleAverageKwargs, *theta: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        values, du_d_theta = BaseOp._compute_base_observables(kwargs, *theta)

        total_mass = _compute_mass(kwargs["system"])
        density = total_mass / values["volume"] * _DENSITY_CONVERSION
        values["density"] = density

        # return a flat list of tensors and the column names
        # as pytorch doesn't currently work with arbitrary pytrees
        # see https://github.com/pytorch/pytorch/issues/96337

        column_names = sorted(values.keys())
        stacked_values = torch.stack([values[name] for name in column_names], dim=0)
        avg_values = stacked_values.mean(dim=0)
        avg_stds = stacked_values.std(dim=0)

        # modify context for backward pass
        ctx.n_theta = len(theta)
        ctx.columns = column_names
        ctx.beta = kwargs["beta"]
        ctx.save_for_backward(
            *theta,
            *du_d_theta,
            stacked_values,
            avg_values
        )
        ctx.mark_non_differentiable(*avg_stds)
        return *avg_values, *avg_stds, tuple(column_names)


    @staticmethod
    def backward(ctx, *grad_outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor | None, ...]:

        # TODO: follow same pattern as osmotic_coefficient, can this be generalized?
        raise NotImplementedError("Keep going.")
