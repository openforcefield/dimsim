from collections.abc import Iterable
from typing import TypeAlias

from openff.toolkit.topology.molecule import Molecule

MoleculeLike: TypeAlias = Molecule

class Topology:
    @classmethod
    def from_molecules(cls, molecules: MoleculeLike | Iterable[MoleculeLike]) -> Topology: ...
