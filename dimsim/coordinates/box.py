"""
Coordinate database for storing pre-equilibrated molecular simulation boxes.

This module provides a SQLite-based storage system for molecular simulation
coordinates with searchable metadata including SMILES, thermodynamic states,
force fields, and energies.
"""

import datetime
import gzip
import json
import typing
import warnings

import descent.targets.thermo
import numpy as np
import openff.toolkit
import openmm
import pydantic
from openff.toolkit import Molecule
from openff.toolkit.utils.exceptions import MultipleComponentsInMoleculeWarning
from openff.utilities.utilities import requires_package
from rdkit import Chem
from sqlalchemy import LargeBinary
from sqlmodel import Column, Field, SQLModel

from ..base import BaseModel


class MoleculeSpecies(BaseModel):
    """A molecule species in a simulation box."""

    mapped_smiles: str
    count: int

    def to_inchi_composition(self) -> tuple[str, int]:
        """
        Convert mapped SMILES to InChI string.
        These are used for comparison,
        as they're easier to compare than mapped SMILES.

        .. note::

            There may be false positives (=false matches) in inchi comparisons,
            but false negatives should be rare.
            False positives can be deduplicated later through graph isomorphism checks

        Returns
        -------
        tuple[str, int]
            InChI string and count
        """
        mol = Molecule.from_mapped_smiles(self.mapped_smiles)
        inchi = mol.to_inchi(fixed_hydrogens=True)
        return inchi, self.count

    def to_string(self) -> str:
        """
        Get a string representation of the molecule species.

        Returns
        -------
        str
            String representation
        """
        return f"{self.mapped_smiles}<{self.count}>"

    @classmethod
    def from_string(
        cls, s: str, flatten: bool = True
    ) -> "MoleculeSpecies | list[MoleculeSpecies]":
        """
        Create a MoleculeSpecies (or multiple) from its string representation.

        Parameters
        ----------
        s : str
            String representation. If multiple species, separate with '|'
        flatten : bool, default=True
            If True, return a single MoleculeSpecies if only one is present.

        Returns
        -------
        MoleculeSpecies or list[MoleculeSpecies]
            MoleculeSpecies object/s
        """
        fields = s.split("|")
        species = []
        for field in fields:
            # TODO: proper regex validation?
            assert field.endswith(">"), f"Invalid molecule species string: {field}"
            mapped_smiles, count_str = field.rstrip(">").split("<")
            species.append(cls(mapped_smiles=mapped_smiles, count=int(count_str)))

        if flatten and len(species) == 1:
            return species[0]
        return species


class Substance(BaseModel):
    """A substance in a simulation box, consisting of multiple molecule species."""

    molecule_species: list[MoleculeSpecies]

    def __hash__(self) -> int:
        return hash(self.to_composition_key())

    def to_string(self) -> str:
        """
        Get a string representation of the substance.

        Returns
        -------
        str
            String representation
        """
        return "|".join(s.to_string() for s in self.molecule_species)

    def to_composition_key(self) -> str:
        """
        Get a canonical composition key for the substance.

        Returns
        -------
        str
            Canonical composition key
        """
        sorted_species = sorted(
            [s.to_inchi_composition() for s in self.molecule_species]
        )
        return "|".join(f"{inchi}:{count}" for inchi, count in sorted_species)

    @classmethod
    def from_string(cls, s: str) -> "Substance":
        """
        Create a Substance from its string representation.

        Parameters
        ----------
        s : str
            String representation. Separate species with '|'

        Returns
        -------
        Substance
            Substance object
        """
        return cls(molecule_species=MoleculeSpecies.from_string(s, flatten=False))

    @property
    def n_molecules(self) -> int:
        """
        Get the total number of molecules in the substance.

        Returns
        -------
        int
            Total number of molecules
        """
        return sum(s.count for s in self.molecule_species)

    def to_openff_topology(self) -> "openff.toolkit.Topology":
        """
        Convert to an OpenFF Toolkit Topology object.

        Returns
        -------
        openff.toolkit.Topology
            OpenFF Topology representation of the substance
        """
        from openff.toolkit import Molecule, Topology

        molecules = []
        for mol in self.molecule_species:
            molecules.extend(
                [Molecule.from_mapped_smiles(mol.mapped_smiles)] * mol.count
            )
        top = Topology.from_molecules(molecules)
        return top

    @classmethod
    def from_openff_topology(
        cls, topology: "openff.toolkit.Topology"
    ) -> "BoxCoordinates":
        """
        Create BoxCoordinates from an OpenFF Toolkit Topology object.

        Parameters
        ----------
        topology : openff.toolkit.Topology
            OpenFF Topology to convert from

        Returns
        -------
        BoxCoordinates
            BoxCoordinates representation of the topology
        """
        first_appearance_index = sorted(topology.identical_molecule_groups)
        molecule_species = []
        molecules = list(topology.molecules)
        for key in first_appearance_index:
            mol = molecules[key]
            mapped_smiles = mol.to_smiles(mapped=True)
            count = len(topology.identical_molecule_groups[key])
            molecule_species.append(
                MoleculeSpecies(mapped_smiles=mapped_smiles, count=count)
            )

        return cls(molecule_species=molecule_species)

    def is_equivalent_to(self, other: "Substance") -> bool:
        """
        Check if this substance is equivalent to another substance.
        This is done by comparing the composition keys.

        Parameters
        ----------
        other : Substance
            Other substance to compare with

        Returns
        -------
        bool
            True if both substances are equivalent, False otherwise
        """
        if not self.to_composition_key() == other.to_composition_key():
            return False

        ...  # TODO: add graph isomorphism checks?

    @classmethod
    def from_data_entry(cls, entry, n_max_mols: int) -> "Substance":
        smiles_a = entry["smiles_a"]
        smiles_b = entry["smiles_b"]

        fraction_a = 0.0 if entry["x_a"] is None else entry["x_a"]
        fraction_b = 0.0 if entry["x_b"] is None else entry["x_b"]

        assert np.isclose(fraction_a + fraction_b, 1.0)


        smiles_and_fractions = []
        if smiles_a is not None:
            smiles_and_fractions.append((smiles_a, fraction_a))
        if smiles_b is not None:
            smiles_and_fractions.append((smiles_b, fraction_b))

        all_molecule_species = []
        for smiles, fraction in smiles_and_fractions:
            n_molecules = fraction * n_max_mols

            if "." in smiles:  # this is multi-molecule
                # this will give us a MultipleComponentsInMoleculeWarning -- suppress
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=MultipleComponentsInMoleculeWarning
                    )
                    mol = Molecule.from_mapped_smiles(
                        smiles, allow_undefined_stereo=True
                    )

                # use rdkit to break them apart and reformat as smiles
                rdmol = mol.to_rdkit()
                components = Chem.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
                mols = [
                    Molecule.from_rdkit(m, allow_undefined_stereo=True)
                    for m in components
                ]
                component_smiles = [m.to_smiles(mapped=True) for m in mols]
            else:
                component_smiles = [smiles]

            n_components = len(component_smiles)
            for comp_smiles in component_smiles:
                # probably want at least one in there?
                molecule_species = max(1, int(n_molecules / n_components))
                all_molecule_species.append(
                    MoleculeSpecies(mapped_smiles=comp_smiles, count=molecule_species)
                )
        return cls(molecule_species=all_molecule_species)

    @classmethod
    def from_simulation_key(
        cls, key: descent.targets.thermo.SimulationKey
    ) -> "Substance":
        """
        Create Substance from a SimulationKey.

        Parameters
        ----------
        key : descent.targets.thermo.SimulationKey
            SimulationKey to create Substance from
        """

        molecule_species = [
            MoleculeSpecies(mapped_smiles=smiles, count=count)
            for smiles, count in zip(key.smiles, key.counts, strict=True)
        ]
        return cls(molecule_species=molecule_species)



class CoordinatesDB(SQLModel, table=True):
    """Database model for coordinates."""

    __tablename__ = "coordinates"

    id: int | None = Field(
        default=None,
        primary_key=True,
        description="Unique identifier for the coordinate entry",
    )
    composition_key: str = Field(
        index=True,
        description="Canonical composition key (InChI:count|InChI:count|...)",
    )
    molecule_species: str = Field(
        index=True,
        description=(
            "String representation of molecule species "
            "(SMILES<count>|SMILES<count>|...)"
        )
    )
    n_molecules: int = Field(
        index=True, description="Total number of molecules in the box"
    )
    temperature: float | None = Field(index=True, description="Temperature in Kelvin")
    pressure: float | None = Field(index=True, description="Pressure in atm")
    force_field_id: str | None = Field(
        index=True, description="Identifier for the force field used"
    )
    potential_energy: float | None = Field(
        index=True, description="Potential energy of the system in kcal/mol"
    )
    coordinates: bytes = Field(
        sa_column=Column(LargeBinary), description="Compressed binary coordinates"
    )
    box_vectors: bytes | None = Field(
        sa_column=Column(LargeBinary), description="Compressed binary box vectors"
    )
    box_metadata: str = Field(default="{}", description="JSON-encoded metadata")
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp when the box was created",
    )


class BoxCoordinates(BaseModel):
    """
    A set of coordinates for a simulation box.

    This represents a single snapshot of a molecular system with associated
    metadata about composition, thermodynamic state, and force field.
    """

    # Unique identifier (assigned by database)
    id: int | None = None

    # Composition
    substance: Substance

    # Thermodynamic state
    temperature: float | None = pydantic.Field(
        None, description="Temperature in Kelvin"
    )
    pressure: float | None = pydantic.Field(None, description="Pressure in atm")

    # Force field identifier
    force_field_id: str | None = pydantic.Field(
        None, description="Identifier for the force field used"
    )

    # Energy (kcal/mol)
    potential_energy: float | None = pydantic.Field(
        None, description="Potential energy of the system in kcal/mol"
    )

    # Coordinates as numpy array (n_atoms, 3) in angstrom
    # Stored as compressed binary in database
    coordinates: np.ndarray | None = pydantic.Field(
        None, description="Atomic coordinates in angstrom"
    )

    # Box vectors (3, 3) in angstrom
    box_vectors: np.ndarray | None = pydantic.Field(
        None, description="Box vectors in angstrom"
    )

    # typing.Optional metadata (flat-bottom restraints, simulation details, etc.)
    box_metadata: dict[str, typing.Any] = pydantic.Field(
        default_factory=dict, description="Additional metadata about the box"
    )

    # Timestamp
    created_at: datetime.datetime | None = pydantic.Field(
        None, description="Timestamp when the box was created"
    )

    class Config:
        arbitrary_types_allowed = True


    def __hash__(self) -> int:
        return hash(
            (
                self.substance.to_string(),
                self.temperature,
                self.pressure,
                self.force_field_id,
                self.potential_energy,
                (
                    self.coordinates.data.tobytes()
                    if self.coordinates is not None
                    else None
                ),
                (
                    self.box_vectors.data.tobytes()
                    if self.box_vectors is not None
                    else None
                ),
                json.dumps(self.box_metadata, sort_keys=True),
            )
        )

    def to_simulation_key(self) -> descent.targets.thermo.SimulationKey:
        return descent.targets.thermo.SimulationKey(
            smiles=[s.mapped_smiles for s in self.substance.molecule_species],
            counts=[s.count for s in self.substance.molecule_species],
            temperature=self.temperature,
            pressure=self.pressure,
        )

    @classmethod
    def from_simulation_key(
        cls, key: descent.targets.thermo.SimulationKey
    ) -> "BoxCoordinates":
        """
        Create BoxCoordinates from a SimulationKey.

        Parameters
        ----------
        key : descent.targets.thermo.SimulationKey
            SimulationKey to create BoxCoordinates from
        """
        substance = Substance.from_simulation_key(key)
        return cls(
            substance=substance,
            temperature=key.temperature,
            pressure=key.pressure,
        )

    @classmethod
    def from_data_entry(
        cls, entry: descent.targets.thermo.DataEntry,
        n_max_mols: int = 1000
    ) -> "BoxCoordinates":
        """
        Create BoxCoordinates from a DataEntry.

        Parameters
        ----------
        entry : descent.targets.thermo.DataEntry
            Data entry to create BoxCoordinates from
        n_max_mols : int, default=1000
            Maximum number of molecules to use for composition scaling

        Returns
        -------
        BoxCoordinates
            BoxCoordinates representation of the data entry.
            The actual box data (coordinates, vectors etc) will be None.
        """
        substance = Substance.from_data_entry(entry, n_max_mols=n_max_mols)
        return cls(
            substance=substance,
            temperature=entry.get("temperature"),
            pressure=entry.get("pressure"),
        )



    def get_molecule_string(self) -> str:
        """
        Get a string representation of the molecules in the box.
        The format is "SMILES1<count1>|SMILES2<count2>|...".

        Returns
        -------
        str
            String representation of molecules
        """
        return self.substance.to_string()

    def get_composition_key(self) -> str:
        """
        Get a canonical string representation of the box composition.

        This is solely used for quickly identifying boxes with the same composition.
        The output string is a sorted concatenation of InChI strings and counts.
        The format is "InChI1:count1|InChI2:count2|...".

        Returns
        -------
        str
            Canonical composition string.


        """
        return self.substance.to_composition_key()

    @property
    def n_molecules(self) -> int:
        """
        Get the total number of molecules in the box.

        Returns
        -------
        int
            Total number of molecules
        """
        return self.substance.n_molecules

    def to_openff_topology(self) -> "openff.toolkit.Topology":
        """
        Convert to an OpenFF Toolkit Topology object.

        Returns
        -------
        openff.toolkit.Topology
            OpenFF Topology representation of the box
        """
        top = self.substance.to_openff_topology()
        if self.coordinates is not None:
            top.set_positions(self.coordinates * openff.toolkit.unit.angstrom)
        if self.box_vectors is not None:
            top.box_vectors = self.box_vectors * openff.toolkit.unit.angstrom
        return top

    @classmethod
    def from_openff_topology(
        cls, topology: "openff.toolkit.Topology"
    ) -> "BoxCoordinates":
        """
        Create BoxCoordinates from an OpenFF Toolkit Topology object.

        Parameters
        ----------
        topology : openff.toolkit.Topology
            OpenFF Topology to convert from

        Returns
        -------
        BoxCoordinates
            BoxCoordinates representation of the topology
        """
        from openff.toolkit import unit

        substance = Substance.from_openff_topology(topology)

        coordinates = topology.get_positions()
        if coordinates is not None:
            coordinates = coordinates.m_as(unit.angstrom)

        box_vectors = topology.box_vectors
        if box_vectors is not None:
            box_vectors = box_vectors.m_as(unit.angstrom)

        return cls(
            substance=substance, coordinates=coordinates, box_vectors=box_vectors
        )

    @requires_package("openmm")
    def get_energy_for_system(self, system: "openmm.System") -> float:
        """
        Get the potential energy of the box.

        Parameters
        ----------
        system : openmm.System
            OpenMM System object to calculate energy for

        Returns
        -------
        float
            Potential energy in kcal/mol
        """
        ...

    def has_equivalent_molecular_species(self, other: "BoxCoordinates") -> bool:
        """
        Check if this box has the same molecular species and counts as another box.
        This first compares the composition keys (InChI + count),
        and then the actual molecular graphs to avoid false positives.

        Parameters
        ----------
        other : BoxCoordinates
            Other BoxCoordinates object to compare with

        Returns
        -------
        bool
            True if both boxes have the same molecular species and counts
        """
        return self.substance.is_equivalent_to(other.substance)

    def load_coordinates_from_other(self, other: "BoxCoordinates") -> None:
        """
        Load coordinates and box vectors from another BoxCoordinates object.
        This is loaded in place.

        Parameters
        ----------
        other : BoxCoordinates
            Other BoxCoordinates object to load from
        """

        assert other.coordinates is not None, "Other box must have coordinates to load"

        # Load coordinates, accounting for remappings
        ...

    def with_coordinates_from_other(self, other: "BoxCoordinates") -> "BoxCoordinates":
        """
        Create a new BoxCoordinates object with coordinates and box vectors
        loaded from another BoxCoordinates object.

        Parameters
        ----------
        other : BoxCoordinates
            Other BoxCoordinates object to load from

        Returns
        -------
        BoxCoordinates
            New BoxCoordinates object with loaded coordinates
        """
        assert other.coordinates is not None, "Other box must have coordinates to load"

        new_box = self.model_copy(deep=True)
        new_box.load_coordinates_from_other(other)
        return new_box

    def _to_db_model(self) -> CoordinatesDB:
        """
        Convert to database model for storage.

        Returns
        -------
        CoordinatesDB
            Database model object
        """
        assert (
            self.coordinates is not None
        ), "Coordinates must be set to store in database"
        assert (
            self.box_vectors is not None
        ), "Box vectors must be set to store in database"

        coords_bytes = gzip.compress(self.coordinates.tobytes())
        box_vectors_bytes = (
            gzip.compress(self.box_vectors.tobytes())
            if self.box_vectors is not None
            else None
        )

        return CoordinatesDB(
            composition_key=self.substance.to_composition_key(),
            molecule_species=self.substance.to_string(),
            n_molecules=self.n_molecules,
            temperature=self.temperature,
            pressure=self.pressure,
            force_field_id=self.force_field_id,
            potential_energy=self.potential_energy,
            coordinates=coords_bytes,
            box_vectors=box_vectors_bytes,
            box_metadata=json.dumps(self.box_metadata),
        )

    @classmethod
    def _from_db_model(cls, db_model: CoordinatesDB) -> "BoxCoordinates":
        """
        Create BoxCoordinates from database model.

        Parameters
        ----------
        db_model : CoordinatesDB
            Database model object

        Returns
        -------
        BoxCoordinates
            BoxCoordinates object
        """
        # Get molecule species
        substance = Substance.from_string(db_model.molecule_species)

        # Decompress coordinates
        coords = np.frombuffer(
            gzip.decompress(db_model.coordinates), dtype=np.float64
        ).reshape(-1, 3)

        box_vectors = np.frombuffer(
            gzip.decompress(db_model.box_vectors), dtype=np.float64
        ).reshape(3, 3)

        obj = cls(
            id=db_model.id,
            substance=substance,
            temperature=db_model.temperature,
            pressure=db_model.pressure,
            force_field_id=db_model.force_field_id,
            potential_energy=db_model.potential_energy,
            coordinates=coords,
            box_vectors=box_vectors,
            box_metadata=(
                json.loads(db_model.box_metadata) if db_model.box_metadata else {}
            ),
            created_at=db_model.created_at,
        )
        assert obj.n_molecules == db_model.n_molecules

        # TODO: validate composition key, and warn if it mismatches
        return obj

    @staticmethod
    def get_lowest_energy_box(boxes: list["BoxCoordinates"]) -> "BoxCoordinates":
        """
        Get the BoxCoordinates object with the lowest potential energy
        from a list of boxes.

        Parameters
        ----------
        boxes : list[BoxCoordinates]
            List of BoxCoordinates objects to search

        Returns
        -------
        BoxCoordinates
            BoxCoordinates object with the lowest potential energy
        """
        if any(box.potential_energy is None for box in boxes):
            raise ValueError(
                "All boxes must have potential energy defined to find the "
                "lowest energy box."
            )
        return min(boxes, key=lambda box: box.potential_energy)
