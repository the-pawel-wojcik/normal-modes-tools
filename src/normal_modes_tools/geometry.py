from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import xyz_parser as xyz

@dataclass
class AtomVector:
    name: str
    xyz: list[float]


@dataclass
class Geometry:
    atoms: list[AtomVector]
    point_group: str = ""

    def to_numpy(self) -> NDArray[np.float64]:
        dim = 3 * len(self.atoms)
        geometry = np.zeros(shape=dim)
        for idx, atom in enumerate(self.atoms):
            geometry[3 * idx: 3 * idx + 3] = [
                np.float64(cart) for cart in atom.xyz
            ]

        return geometry

    def get_mass_matrix(
        self, masses_dict: dict[str, float]
    ) -> NDArray[np.float64]:
        dim = 3 * len(self.atoms)
        diagonal = np.zeros(shape=dim, dtype=np.float64)
        for idx, atom in enumerate(self.atoms):
            atom_mass = np.float64(masses_dict[atom.name.capitalize()])
            diagonal[3 * idx: 3 * idx + 3] = [atom_mass for _ in range(3)]
        mass_matrix = np.zeros(shape=(dim, dim), dtype=np.float64)
        np.fill_diagonal(mass_matrix, diagonal)
        return mass_matrix
    

    @classmethod
    def from_numpy(cls, vec, atom_names):
        assert len(vec) % 3 == 0
        assert len(vec) // 3 == len(atom_names)
        atoms = list()
        for idx, element in enumerate(atom_names):
            atoms.append(AtomVector(
                name=element,
                xyz=[float(coord) for coord in vec[3*idx: 3*idx+3]]
            ))
        return cls(atoms=atoms)


    @classmethod
    def from_MoleculeXYZ(cls, molecule: xyz.MoleculeXYZ) -> Geometry:
        """ TODO: This is too similar to NormalMode.from_MoleculeXYZ """
        geometry = Geometry(atoms=list())
        for atom in molecule.atoms:
            geometry.atoms.append(AtomVector(
                name = atom.symbol,
                xyz = [atom.x, atom.y, atom.z],
            ))
        return geometry

    def xyz_str(self, comment: str = "comment") -> str:
        """ A drop-in replacement for xyz.MoleculeXYZ.from_Geometry. """
        xyz = f'{len(self.atoms)}\n'
        xyz += f'{comment}\n'
        for atom in self.atoms:
            xyz += f'{atom.name:<8}'
            for coordinate in atom.xyz:
                xyz += f'{coordinate:12.6f}'
            xyz += '\n'
        xyz = xyz[:-1]  # remove the trailing newline
        return xyz
