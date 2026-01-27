"""OpenMM simulation reporters"""

import contextlib
import math
import os
import typing

import msgpack
import numpy
import openmm.app
import openmm.unit
import torch

_ANGSTROM = openmm.unit.angstrom
_KCAL_PER_MOL = openmm.unit.kilocalories_per_mole
_KCAL_PER_MOL_PER_ANGSTROM = openmm.unit.kilocalories_per_mole / openmm.unit.angstrom


def _encoder(obj, chain=None):
    """msgpack encoder for tensors"""
    if isinstance(obj, torch.Tensor):
        assert obj.dtype == torch.float32
        return {b"torch": True, b"shape": obj.shape, b"data": obj.numpy().tobytes()}
    else:
        return obj if chain is None else chain(obj)


def _decoder(obj, chain=None):
    """msgpack decoder for tensors"""
    try:
        if b"torch" in obj:
            array = numpy.ndarray(
                buffer=obj[b"data"], dtype=numpy.float32, shape=obj[b"shape"]
            )
            return torch.from_numpy(array.copy())
        else:
            return obj if chain is None else chain(obj)
    except KeyError:
        return obj if chain is None else chain(obj)


class SimulationTensorReporter:
    """A reporter which stores coords, box vectors, reduced potentials, kinetic
    energy, and custom forces using msgpack.

    TODO: Replaces/extends smee.mm._reporters.TensorReporter and possibly should be upstreamed.
    """

    def __init__(
        self,
        output_file: typing.BinaryIO,
        report_interval: int,
        beta: openmm.unit.Quantity,
        pressure: openmm.unit.Quantity | None,
        custom_force_groups: dict[str, int] | None = None,
    ):
        """

        Args:
            output_file: The file to write the frames to.
            report_interval: The interval (in steps) at which to write frames.
            beta: The inverse temperature the simulation is being run at.
            pressure: The pressure the simulation is being run at, or None if NVT /
                vacuum.
        """
        self._output_file = output_file
        self._report_interval = report_interval

        self._beta = beta
        self._pressure = (
            None if pressure is None else pressure * openmm.unit.AVOGADRO_CONSTANT_NA
        )
        if custom_force_groups is None:
            custom_force_groups = {}
        self.custom_force_groups = dict(custom_force_groups)
        self._id_to_name = {v: k for k, v in self.custom_force_groups.items()}

    def describeNextReport(self, simulation: openmm.app.Simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        # requires - positions, velocities, forces, energies?
        return steps, True, False, False, True

    def report(self, simulation: openmm.app.Simulation, state: openmm.State):
        potential_energy = state.getPotentialEnergy()
        kinetic_energy = state.getKineticEnergy()

        total_energy = potential_energy + kinetic_energy

        if math.isnan(total_energy.value_in_unit(_KCAL_PER_MOL)):
            raise ValueError("total energy is nan")
        if math.isinf(total_energy.value_in_unit(_KCAL_PER_MOL)):
            raise ValueError("total energy is infinite")

        unreduced_potential = potential_energy

        if self._pressure is not None:
            unreduced_potential += self._pressure * state.getPeriodicBoxVolume()

        reduced_potential = unreduced_potential * self._beta

        coords = state.getPositions(asNumpy=True).value_in_unit(_ANGSTROM)
        coords = torch.from_numpy(coords).float()
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(_ANGSTROM)
        box_vectors = torch.from_numpy(box_vectors).float()

        # add custom forces
        all_custom_forces = []
        sorted_force_groups = sorted(self.custom_force_groups.values())
        for force_group in sorted_force_groups:
            custom_state = simulation.context.getState(
                getForces=True, groups={force_group}
            )
            custom_forces = torch.from_numpy(
                custom_state.getForces(asNumpy=True).value_in_unit(
                    _KCAL_PER_MOL_PER_ANGSTROM
                )
            ).float()
            all_custom_forces.append(custom_forces)
        frame = (
            coords,
            box_vectors,
            reduced_potential,
            kinetic_energy.value_in_unit(_KCAL_PER_MOL),
            *all_custom_forces,
        )
        self._output_file.write(msgpack.dumps(frame, default=_encoder))

    @staticmethod
    def unpack_frames(
        file: typing.BinaryIO,
    ) -> typing.Generator[tuple[torch.Tensor, torch.Tensor, float, float], None, None]:
        """
        Unpack frames saved by a SimulationTensorReporter.

        This is a generator that yields tuples of tensors in the order
        (coords, box_vectors, reduced_potential, kinetic_energy).
        If custom forces were saved, they will be returned as additional tensors in the tuple,
        after the kinetic_energy tensor, in the order they were specified in the custom_force_groups list.

        """
        unpacker = msgpack.Unpacker(file, object_hook=_decoder)

        for frame in unpacker:
            yield frame

    def unpack_frames_as_dict(self):
        """
        Unpack frames saved by a SimulationTensorReporter as a dictionary of tensors.

        This is a generator that yields dictionaries with keys 'coords', 'box_vectors', 'reduced_potential', 'kinetic_energy',
        and any custom forces specified in the custom_force_groups list.
        """

        index_to_id = zip(enumerate(self._id_to_name))

        # assert this was opened in read-binary mode
        if not self._output_file.readable() or 'b' not in self._output_file.mode:
            raise ValueError("Output file must be opened in read-binary mode")

        unpacker = msgpack.Unpacker(self._output_file, object_hook=_decoder)

        for frame in unpacker:
            coords, box_vectors, reduced_potential, kinetic_energy, *custom_forces = frame
            framedict = {
                "coordinates": coords,
                "box_vectors": box_vectors,
                "reduced_potential": reduced_potential,
                "kinetic_energy": kinetic_energy,
            }
            for index, forces in enumerate(custom_forces):
                group_id = index_to_id[index]
                framedict[self._id_to_name[group_id]] = forces
            yield framedict




@contextlib.contextmanager
def tensor_reporter(
    output_path: os.PathLike,
    report_interval: int,
    beta: openmm.unit.Quantity,
    pressure: openmm.unit.Quantity | None,
    custom_force_groups: dict[str, int] | None = None,
) -> typing.Generator[SimulationTensorReporter, None, None]:
    """Create a ``TensorReporter`` capable of writing frames to a file.

    Args:
        output_path: The path to write the frames to.
        report_interval: The interval (in steps) at which to write frames.
        beta: The inverse temperature the simulation is being run at.
        pressure: The pressure the simulation is being run at, or ``None`` if NVT /
            vacuum.
        custom_force_groups: A dictionary mapping custom force group names to their integer IDs.
    """
    with open(output_path, "wb") as output_file:
        reporter = SimulationTensorReporter(output_file, report_interval, beta, pressure, custom_force_groups=custom_force_groups)
        yield reporter
