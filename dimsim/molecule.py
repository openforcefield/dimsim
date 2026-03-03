from openff.toolkit import Molecule


def map_smiles(smiles: str) -> str:
    molecule: Molecule = Molecule.from_smiles(smiles)
    mapped_smiles: str = molecule.to_smiles(mapped=True)  # type: ignore[return-value]

    return mapped_smiles
