"""Valence potential energy functions."""

import torch

import dimsim.geometry
import dimsim.potentials
import dimsim.utils


@dimsim.potentials.potential_energy_fn(dimsim.PotentialType.BONDS, dimsim.EnergyFn.BOND_HARMONIC)
def compute_harmonic_bond_energy(
    system: dimsim.TensorSystem,
    potential: dimsim.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of bonds for a given
    conformer using a harmonic potential of the form ``1/2 * k * (r - length) ** 2``

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """

    parameters = dimsim.potentials.broadcast_parameters(system, potential)
    particle_idxs = dimsim.potentials.broadcast_idxs(system, potential)

    _, distances = dimsim.geometry.compute_bond_vectors(conformer, particle_idxs)

    k = parameters[:, potential.parameter_cols.index("k")]
    length = parameters[:, potential.parameter_cols.index("length")]

    return (0.5 * k * (distances - length) ** 2).sum(-1)


@dimsim.potentials.potential_energy_fn(dimsim.PotentialType.ANGLES, dimsim.EnergyFn.ANGLE_HARMONIC)
def compute_harmonic_angle_energy(
    system: dimsim.TensorSystem,
    potential: dimsim.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of valence angles for a given
    conformer using a harmonic potential of the form ``1/2 * k * (theta - angle) ** 2``

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """

    parameters = dimsim.potentials.broadcast_parameters(system, potential)
    particle_idxs = dimsim.potentials.broadcast_idxs(system, potential)

    theta = dimsim.geometry.compute_angles(conformer, particle_idxs)

    k = parameters[:, potential.parameter_cols.index("k")]
    angle = parameters[:, potential.parameter_cols.index("angle")]

    return (0.5 * k * (theta - angle) ** 2).sum(-1)


def _compute_cosine_torsion_energy(
    system: dimsim.TensorSystem,
    potential: dimsim.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of torsions for a given
    conformer using a cosine potential of the form
    ``k/idivf*(1+cos(periodicity*phi-phase))``

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """

    parameters = dimsim.potentials.broadcast_parameters(system, potential)
    particle_idxs = dimsim.potentials.broadcast_idxs(system, potential)

    phi = dimsim.geometry.compute_dihedrals(conformer, particle_idxs)

    k = parameters[:, potential.parameter_cols.index("k")]
    periodicity = parameters[:, potential.parameter_cols.index("periodicity")]
    phase = parameters[:, potential.parameter_cols.index("phase")]
    idivf = parameters[:, potential.parameter_cols.index("idivf")]

    return ((k / idivf) * (1.0 + torch.cos(periodicity * phi - phase))).sum(-1)


@dimsim.potentials.potential_energy_fn(dimsim.PotentialType.PROPER_TORSIONS, dimsim.EnergyFn.TORSION_COSINE)
def compute_cosine_proper_torsion_energy(
    system: dimsim.TensorSystem,
    potential: dimsim.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of proper torsions
    for a given conformer using a cosine potential of the form:

    `k*(1+cos(periodicity*theta-phase))`

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """
    return _compute_cosine_torsion_energy(system, potential, conformer)


@dimsim.potentials.potential_energy_fn(dimsim.PotentialType.IMPROPER_TORSIONS, dimsim.EnergyFn.TORSION_COSINE)
def compute_cosine_improper_torsion_energy(
    system: dimsim.TensorSystem,
    potential: dimsim.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of improper torsions
    for a given conformer using a cosine potential of the form:

    `k*(1+cos(periodicity*theta-phase))`

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """
    return _compute_cosine_torsion_energy(system, potential, conformer)


@dimsim.potentials.potential_energy_fn(dimsim.PotentialType.LINEAR_BONDS, dimsim.EnergyFn.BOND_LINEAR)
def compute_linear_bond_energy(
    system: dimsim.TensorSystem,
    potential: dimsim.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of bonds for a given
    conformer using a linearized harmonic potential of the form
    ``1/2 * (k1+k2) * (r - (k1 * b1 + k2 * b2) / k) ** 2``

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """
    parameters = dimsim.potentials.broadcast_parameters(system, potential)
    particle_idxs = dimsim.potentials.broadcast_idxs(system, potential)

    _, distances = dimsim.geometry.compute_bond_vectors(conformer, particle_idxs)

    k1 = parameters[:, potential.parameter_cols.index("k1")]
    k2 = parameters[:, potential.parameter_cols.index("k2")]
    b1 = parameters[:, potential.parameter_cols.index("b1")]
    b2 = parameters[:, potential.parameter_cols.index("b2")]
    k0 = k1 + k2
    b0 = (k1 * b1 + k2 * b2) / k0
    return (0.5 * k0 * (distances - b0) ** 2).sum(-1)


@dimsim.potentials.potential_energy_fn(dimsim.PotentialType.LINEAR_ANGLES, dimsim.EnergyFn.ANGLE_LINEAR)
def compute_linear_angle_energy(
    system: dimsim.TensorSystem,
    potential: dimsim.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of valence angles for a given
        conformer using a linearized harmonic potential of the form
    ``1/2 * (k1+k2) * (r - (k1 * angle1 + k2 * angle2) / k) ** 2``

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """

    parameters = dimsim.potentials.broadcast_parameters(system, potential)
    particle_idxs = dimsim.potentials.broadcast_idxs(system, potential)
    theta = dimsim.geometry.compute_angles(conformer, particle_idxs)
    k1 = parameters[:, potential.parameter_cols.index("k1")]
    k2 = parameters[:, potential.parameter_cols.index("k2")]
    a1 = parameters[:, potential.parameter_cols.index("angle1")]
    a2 = parameters[:, potential.parameter_cols.index("angle2")]
    k0 = k1 + k2
    a0 = (k1 * a1 + k2 * a2) / k0
    return (0.5 * k0 * (theta - a0) ** 2).sum(-1)
