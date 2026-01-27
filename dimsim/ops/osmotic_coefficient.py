import torch

from .base import BaseOp, _OpKwargs
from .utils import _unpack_context_and_outputs


class OsmoticCoefficientOp(BaseOp):
    """
    PyTorch operation to compute the osmotic pressure from simulation data.
    """

    @staticmethod
    def forward(ctx, kwargs: _OpKwargs, *theta: torch.Tensor) -> tuple[torch.Tensor, ...]:
        config = kwargs["property_config"]
        values, du_d_theta = BaseOp._compute_base_observables(kwargs, *theta)

        # look for the specified force
        force_name = config.force_name
        assert force_name in values, (
            f"Osmotic pressure requires force '{force_name}' to be computed."
        )
        force_values = values[force_name]

        # compute osmotic pressure
        axis = config.axis
        axis_index = "xyz".index(axis)
        axis_length = values["box_vectors"][:, axis_index, axis_index]
        wall_area = values["volume"] / axis_length  # A = V / L
        # π = |F| / (Aw1 + Aw2)
        osmotic_pressure = torch.abs(force_values) / (wall_area * 2)

        ideal_pressure = ...  # vMRT
        osmotic_coefficient = osmotic_pressure / ideal_pressure


        # TODO: check units
        osmotic_values = torch.stack([osmotic_pressure, osmotic_coefficient], dim=1)
        means = torch.mean(osmotic_values, dim=0)
        stds = torch.std(osmotic_values, dim=0)
        columns = ["osmotic_pressure", "osmotic_coefficient"]

        # save for backward pass
        ctx.n_theta = len(theta)
        ctx.beta = kwargs["beta"]
        ctx.columns = columns
        ctx.save_for_backward(*theta, *du_d_theta, osmotic_values, means)
        ctx.mark_non_differentiable(*stds)
        return *means, *stds, tuple(columns)





    @staticmethod
    def backward(ctx, *grad_outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor | None, ...]:
        # retrieve saved tensors and only keep gradients for observables
        theta, du_d_theta, values, avg_values, grad_outputs = _unpack_context_and_outputs(
            ctx, grad_outputs
        )

        grads = [None] * len(theta)

        for i in range(len(du_d_theta)):
            # ∂U/∂θ_i: shape (n_parameters, n_parameter_cols, n_frames)
            du_d_theta_i = du_d_theta[i]
            if du_d_theta_i is None:
                continue

            # see Wang, Martinez and Pande 2014, equation 2
            # the derivative of an ensemble average resembles a fluctuation property
            # we are overall going for
            # ∂<A>/∂θ_i = <∂A/∂θ_i> - β ( <A ∂U/∂θ_i> - <A><∂U/∂θ_i> )

            # <∂U/∂θ_i>: shape (n_parameters, n_parameter_cols)
            avg_du_d_theta_i = du_d_theta_i.mean(dim=-1)

            # density doesn't explicitly depend on the parameters
            # so ∂A/∂θ_i = 0
            # we keep it here for completeness, but it will be zero
            # shape (n_parameters, n_parameter_cols)
            avg_dA_d_theta_i = torch.zeros_like(avg_du_d_theta_i)
            # broadcast to shape (n_parameters, n_parameter_cols, 1)
            avg_dA_d_theta_i = avg_dA_d_theta_i[:, :, None]

            # <A ∂U/∂θ_i>
            # du_d_theta_i[:, :, None, :] has shape (n_parameters, n_parameter_cols, 1, n_frames)
            # values.T[None, None, :, :] has shape (1, 1, n_observables, n_frames)
            avg_A_du_d_theta_i = torch.mean(
                du_d_theta_i[:, :, None, :] * values.T[None, None, :, :],
                dim=1
            )

            # <A><∂U/∂θ_i>
            avg_A = avg_values[:, None, None]  # shape (n_observables, 1, 1)
            # shape (n_observables, n_parameters, n_parameter_cols)
            avg_A_avg_du_d_theta_i = avg_A * avg_du_d_theta_i[:, :, None]

            # shape (n_parameters, n_parameter_cols, n_observables)
            d_avg_A_d_theta_i = avg_dA_d_theta_i - ctx.beta * (
                avg_A_du_d_theta_i - avg_A_avg_du_d_theta_i
            )
            grads[i] = d_avg_A_d_theta_i @ grad_outputs

        grads = [
            None if v is None else (v if t.ndim == 2 else v.squeeze(0))
            for t, v in zip(theta, grads, strict=True)
        ]

        # we need to return one extra 'gradient' for kwargs.
        return tuple([None] + grads)
