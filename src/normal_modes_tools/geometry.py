from __future__ import annotations
from dataclasses import dataclass
import xyz_parser as xyz
import numpy as np

@dataclass
class AtomVector:
    name: str
    xyz: list[float]


@dataclass
class Geometry:
    atoms: list[AtomVector]
    point_group: str = ""

    def to_numpy(self) -> np.typing.NDArray[np.float64]:
        dim = 3 * len(self.atoms)
        geometry = np.zeros(shape=dim)
        for idx, atom in enumerate(self.atoms):
            geometry[3 * idx: 3 * idx + 3] = [
                np.float64(cart) for cart in atom.xyz
            ]

        return geometry
    

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
