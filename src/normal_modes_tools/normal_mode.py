from __future__ import annotations
from normal_modes_tools.geometry import Geometry, AtomVector
import xyz_parser as xyz

from copy import deepcopy
from dataclasses import dataclass
import numpy as np

@dataclass
class NormalMode:
    """ Data structure for a storage of an eigenvector of a mass-weighted
    Hessian. """
    frequency: float
    displacement: list[AtomVector]
    at: Geometry
    atomic_masses: list[float] | None = None
    irrep: str = ""

    def __str__(self) -> str:
        fmt = '-13.8f'
        str_xyz = f"{len(self.displacement)}\n"
        str_xyz += f"XX. Vibration mode, {self.frequency:.2f} cm-1,"
        if self.irrep == "":
            str_xyz += f' ???\n'  # TODO: use posym if symmetry is unknown
        else:
            str_xyz += f' {self.irrep}\n'
        for geo, nmd in zip(self.at.atoms, self.displacement):
            assert geo.name == nmd.name
            str_xyz += f"{geo.name:<3}"
            for coord in geo.xyz:
                str_xyz += f"{coord:{fmt}}"
            for coord in nmd.xyz:
                str_xyz += f"{coord:{fmt}}"
            str_xyz += '\n'
        str_xyz = str_xyz[:-1]  # trim trailing new line

        return str_xyz

    def to_numpy(self) -> np.typing.NDArray[np.float64]:
        array = np.zeros(shape=3 * len(self.displacement), dtype=np.float64)
        for idx, atom in enumerate(self.displacement):
            array[3 * idx: 3 * idx + 3] = deepcopy(atom.xyz)
        return array

    def __lt__(self, other: NormalMode) -> bool:
        return self.frequency < other.frequency

    def __gt__(self, other: NormalMode) -> bool:
        return self.frequency > other.frequency


    @classmethod
    def from_numpy(
        cls,
        frequency: float,
        vector: np.typing.NDArray[np.float64],
        geometry: Geometry,
        atomic_masses_list: list[float] | None = None,
    ) -> NormalMode:
        """ `atomic_masses_list` is used for isotope-substituted molecules """

        displacement = deepcopy(geometry.atoms)
        for idx, atom in enumerate(displacement):
            atom.xyz = [float(x) for x in vector[3 * idx: 3 * idx + 3]]

        return NormalMode(
            frequency=frequency,
            displacement=displacement,
            at=geometry,
            atomic_masses=atomic_masses_list,
        )


    def __matmul__(self, other: NormalMode) -> float:
        if not isinstance(other, NormalMode):
            return NotImplemented
        
        left = self.to_numpy()
        left /= np.linalg.norm(left)

        right = other.to_numpy()
        right /= np.linalg.norm(right)

        inner = np.dot(left, right)
        return inner

    def to_MoleculeXYZ(self, comment: str | None = None) -> xyz.MoleculeXYZ:
        """ Comment should look like this:
             27. Vibration mode, 1683.54 cm-1, A1
        """
        if comment is None:
            comment = f"N. Vibrational mode, {self.frequency:2.f} cm-1, sym"
        len_atoms = len(self.at.atoms)
        atoms = list()
        for geo, mode in zip(self.at.atoms, self.displacement):
            atoms.append(xyz.AtomLineXYZ(
                symbol=geo.name,
                x=geo.xyz[0],
                y=geo.xyz[1],
                z=geo.xyz[2],
                extra=mode.xyz,
            ))
        return xyz.MoleculeXYZ(
            natoms=len_atoms,
            comment=comment,
            atoms=atoms,
        )


    @classmethod
    def from_MoleculeXYZ(cls, molecule: xyz.MoleculeXYZ) -> NormalMode:
        try:
            # TODO: The files that I use right now keep frequencies as fourth
            # entry in the comment. This should be improved.
            frequency = float(molecule.comment.split()[3])
        except (ValueError, IndexError) as _:
            raise ValueError(
                "Expecting the fourth element in the xyz comment line to store"
                " the harmonic frequency in cm-1."
            ) from None
            
        # HACK: this is no convention 
        irrep = molecule.comment.split()[-1]
        # HACK: swap them beacuse my input uses the wrong convention
        if irrep == 'B1':
            irrep = 'B2'

        elif irrep == 'B2':
            irrep = 'B1'

        geometry = Geometry(atoms=list())
        normalmode = NormalMode(
            frequency=frequency,
            displacement=list(),
            at=geometry,
            irrep=irrep,
        )

        for atom in molecule.atoms:
            geometry.atoms.append(AtomVector(
                name = atom.symbol,
                xyz = [atom.x, atom.y, atom.z],
            ))
            if len(atom.extra) != 3:
                raise ValueError(
                    "Expecting that each atom line in the xyz file stores six: "
                    "floats: three for the atom's postion, and three for the "
                    "normal mode's dispacement."
                ) from None
            normalmode.displacement.append(AtomVector(
                name = atom.symbol,
                xyz = atom.extra,
            ))
        return normalmode


def moleculeXYZList_to_NormalModesList(
    xyz_nmodes: list[xyz.MoleculeXYZ]
) -> list[NormalMode]:
    normal_modes: list[NormalMode] = list()
    for molecule in xyz_nmodes:
        normal_modes.append(NormalMode.from_MoleculeXYZ(molecule))

    return normal_modes


def xyz_file_to_NormalModesList(
    xyz_path: str,
) -> list[NormalMode]:
    nmodes_xyz = xyz.read_xyz_file(xyz_path)
    nmodes = moleculeXYZList_to_NormalModesList(nmodes_xyz)
    return nmodes


def normalModesList_to_xyz_file(
    nml: list[NormalMode],
) -> str:
    return "\n".join(str(mode) for mode in nml)
