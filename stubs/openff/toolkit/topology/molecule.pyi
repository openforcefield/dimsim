from typing import TypeAlias, TypeVar, Union

from openff.toolkit.utils.toolkits import ToolkitRegistry, ToolkitWrapper

TKR: TypeAlias = ToolkitRegistry | ToolkitWrapper
MoleculeLike: TypeAlias = Union["Molecule", "FrozenMolecule"]
FM = TypeVar("FM", bound="FrozenMolecule")
P = TypeVar("P", bound="Particle")
A = TypeVar("A", bound="Atom")
B = TypeVar("B", bound="Bond")

class Particle:
    pass

class Atom(Particle):
    pass

class Bond:
    pass

class FrozenMolecule:
    def __init__(
        self,
        other=None,
        file_format: str | None = None,
        toolkit_registry: TKR = ...,
        allow_undefined_stereo: bool = False,
    ) -> None: ...

class Molecule(FrozenMolecule):
    @classmethod
    def from_mapped_smiles(cls, smiles: str) -> Molecule: ...
    @classmethod
    def from_smiles(
        cls: type[FM],
        smiles: str,
        hydrogens_are_explicit: bool = False,
        toolkit_registry: TKR = ...,
        allow_undefined_stereo: bool = False,
        name: str = "",
    ) -> FM: ...
    def to_smiles(
        self,
        isomeric: bool = True,
        explicit_hydrogens: bool = True,
        mapped: bool = False,
        toolkit_registry: TKR = ...,
    ) -> str: ...
    def to_inchi(self, fixed_hydrogens: bool = False, toolkit_registry: TKR = ...) -> str: ...
    @classmethod
    def from_iupac(
        cls, iupac_name: str, toolkit_registry: TKR = ..., allow_undefined_stereo: bool = False, **kwargs
    ) -> FM: ...
    def to_iupac(self, toolkit_registry=...): ...
