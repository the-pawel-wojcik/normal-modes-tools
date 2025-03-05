from __future__ import annotations
import xyz_parser as xyz
from dataclasses import dataclass
from normal_modes_tools.atomic_masses import *
import numpy as np
from numpy import linalg
from copy import deepcopy
import matplotlib.pyplot as plt

@dataclass
class AtomVector:
    name: str
    xyz: list[float]


@dataclass
class Geometry:
    atoms: list[AtomVector]

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


@dataclass
class NormalMode:
    """ Data structure for a storage of an eigenvector of a mass-weighted
    Hessian. """
    frequency: float
    displacement: list[AtomVector]
    at: Geometry
    atomic_masses: list[float] | None = None

    def __str__(self) -> str:
        fmt = '-13.8f'
        str_xyz = f"{len(self.displacement)}\n"
        str_xyz += f"XX. Vibration mode, {self.frequency:.2f} cm-1, SYM\n"
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
            
        geometry = Geometry(atoms=list())
        normalmode = NormalMode(
            frequency=frequency,
            displacement=list(),
            at=geometry,
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


def normalize_symbol(original: str) -> str:
   all_caps  = original.capitalize()
   first_cap = all_caps[0] + all_caps[1:].lower()
   return first_cap


def build_mass_matrix(
    geometry: Geometry,
    masses_dict: dict[str, float]
) -> np.typing.NDArray[np.float64]:
    dim = 3 * len(geometry.atoms)
    diagonal = np.zeros(shape=dim, dtype=np.float64)
    for idx, atom in enumerate(geometry.atoms):
        atom_mass = np.float64(masses_dict[normalize_symbol(atom.name)])
        diagonal[3 * idx: 3 * idx + 3] = [atom_mass for _ in range(3)]
    mass_matrix = np.zeros(shape=(dim, dim), dtype=np.float64)
    np.fill_diagonal(mass_matrix, diagonal)
    return mass_matrix


def build_nmodes_matrix(
    normal_modes: list[NormalMode],
) -> np.typing.NDArray[np.float64]:
    """ Eigenvectors of the mass-weighted Hessian form columns. """

    len_nm = len(normal_modes)
    dim = 3 * len(normal_modes[0].at.atoms)
    nmodes_matrix = np.zeros(
        shape=(dim, len_nm),
        dtype=np.float64,
    )

    for column, mode in enumerate(normal_modes):
        for atom_idx, atom in enumerate(mode.displacement):
            for cart_idx in range(3):
                row = atom_idx * 3 + cart_idx
                nmodes_matrix[row, column] = np.float64(atom.xyz[cart_idx])

    return nmodes_matrix


def build_hessian(
    normal_modes: list[NormalMode],
    mass_matrix: np.typing.NDArray[np.float64]
) -> np.typing.NDArray[np.float64]:

    len_nm = len(normal_modes)

    freq_matrix = np.zeros(
        shape=(len_nm, len_nm),
        dtype=np.float64,
    )

    for column, mode in enumerate(normal_modes):
        freq_matrix[column, column] = np.float64(mode.frequency)

    nmodes_matrix = build_nmodes_matrix(normal_modes)

    weighted_D = np.sqrt(mass_matrix) @ nmodes_matrix 
    hessian = weighted_D @ freq_matrix**2 @ weighted_D.transpose()

    return hessian


def get_mass_inv_sqrt(
        mass_matrix: np.typing.NDArray[np.float64]
) -> np.typing.NDArray[np.float64]:

    mass_sqrt = np.sqrt(mass_matrix)
    mass_inv_sqrt = np.zeros(
        shape=mass_sqrt.shape,
        dtype=mass_sqrt.dtype
    )
    diagonal = mass_sqrt.diagonal()
    np.fill_diagonal(mass_inv_sqrt, 1.0 / diagonal)

    return mass_inv_sqrt


def diagonalize_hessian(
    hessian: np.typing.NDArray[np.float64],
    mass_matrix: np.typing.NDArray[np.float64],
) -> np.linalg._linalg.EighResult:

    mass_inv_sqrt = get_mass_inv_sqrt(mass_matrix)
    mass_weighted_hessian = mass_inv_sqrt @ hessian @ mass_inv_sqrt
    eigensystem = linalg.eigh(mass_weighted_hessian)
    return eigensystem


def print_eigenvalues(
    evals: np.typing.NDArray[np.float64],
    show_tr_rot: bool = False,
) -> None:
    """ `evals` is expected to be the value of the `eigenvalues` attribute of
    the return value of np.linalg.eigh function. """
    freqs = [np.sqrt(freq.real) for freq in evals if freq.real > 1]

    if show_tr_rot is True:
        freqs.extend(0.0 for freq in evals if freq.real <= 1)
    
    for idx, freq in enumerate(sorted(freqs)):
        print(f'{idx + 1:3d}: {freq:>4.0f}')


def print_pair_of_eigenvalues(
    left: np.typing.NDArray[np.float64],
    right: np.typing.NDArray[np.float64],
) -> None:
    """ Both `left` and `right` are expected to be the values of the
    `eigenvalues` attribute of the return value of np.linalg.eigh function."""
    left_freqs = [np.sqrt(freq.real) for freq in left if freq.real > 1]
    right_freqs = [np.sqrt(freq.real) for freq in right if freq.real > 1]

    if len(left_freqs) != len(right_freqs):
        raise RuntimeError("Mismatch in the number of frequencies.")
    
    for idx, (left, right) in enumerate(zip(sorted(left_freqs), sorted(right_freqs))):
        print(f'{idx + 1:3d}: {left:>4.0f} {right:>4.0f}')

def str_eigenvalue(eval):
    if eval > 0:
        return f"{np.sqrt(eval):4.0f}"
    else:
        return f"{0}"


def show_nmodes_matrix(eigensystem: np.linalg._linalg.EighResult) -> None:
    """Display a figure showing the eigsystem of the mass-weighted Hessian. """
    _, ax = plt.subplots()
    nmodes = np.matrix(eigensystem.eigenvectors)
    freqs = [str_eigenvalue(eval) for eval in eigensystem.eigenvalues]
    ax.imshow(nmodes, aspect='equal')
    ax.set_xticks([idx for idx, _ in enumerate(freqs)])
    ax.set_xticklabels(freqs)
    ax.tick_params(axis='x', labelrotation=70)
    # ax.tick_params(axis='x', labelsize=6)
    # plt.xticks_labels(frequencies)
    plt.show()


def generated_displaced_geometry(
    which_mode: int,
    displacement: float,
    reference_geometry: Geometry,
    normal_modes: list[NormalMode],
    mass_matrix: np.typing.NDArray[np.float64],
) -> Geometry:
    """ Build a new molecular geometry by displacing the reference geometry by
    `displacement` AA sqrt(amu) along the normal mode number `which_mode` """
    which_mode = 0
    len_nmodes = len(normal_modes)
    
    nmodes_matrix = build_nmodes_matrix(normal_modes)

    displaced_vector_normal_coordinates = np.zeros(shape=(len_nmodes))
    displaced_vector_normal_coordinates[which_mode] = displacement
    displaced_vector_mass_weighted =\
            nmodes_matrix @ displaced_vector_normal_coordinates
    mass_inv_sqrt = get_mass_inv_sqrt(mass_matrix)
    displacement_Descartes =\
            mass_inv_sqrt @ displaced_vector_mass_weighted\
            + reference_geometry.to_numpy()

    atom_names = [atom.name for atom in reference_geometry.atoms]
    displaced = Geometry.from_numpy(
        vec=displacement_Descartes,
        atom_names=atom_names
    )
    return displaced


def esystem_to_NModes(
    esystem: np.linalg._linalg.EighResult,
    geometry: Geometry,
    atomic_masses_dict: dict[str, float] | None = None,
    skip_tr_rot_modes: bool = True,
) -> list[NormalMode]:
    nmodes = list()

    atomic_masses_list = None
    if atomic_masses_dict is not None:
        atomic_masses_list = list()
        for atom in geometry.atoms:
            mass = atomic_masses_dict[normalize_symbol(atom.name)]
            atomic_masses_list.append(deepcopy(mass))


    for evalue, vector in zip(esystem.eigenvalues, esystem.eigenvectors.T):
        if skip_tr_rot_modes and evalue < 0.1:
            continue
        frequency = np.sqrt(evalue)
        nmodes.append(NormalMode.from_numpy(
            frequency=frequency,
            vector=vector,
            geometry=geometry,
            atomic_masses_list=atomic_masses_list,
        ))
    nmodes.sort()

    return nmodes

def deuterate_modes(
    normal_modes: list[NormalMode],
    present_mode: bool = False,
) -> list[NormalMode]:
    equilibrium_Descarte = normal_modes[0].at
    mass_matrix = build_mass_matrix(equilibrium_Descarte, ATOMIC_MASSES)
    hessian = build_hessian(normal_modes, mass_matrix)
    eigensystem = diagonalize_hessian(hessian, mass_matrix) 

    deuterated_masses = deepcopy(ATOMIC_MASSES)
    deuterated_masses['H'] = 2.0
    deuterated_mm = build_mass_matrix(equilibrium_Descarte, deuterated_masses)
    deuterated = diagonalize_hessian(hessian, deuterated_mm)

    deuterated_nmodes = esystem_to_NModes(deuterated, equilibrium_Descarte)

    if present_mode is True:
        print_pair_of_eigenvalues(
            eigensystem.eigenvalues,
            deuterated.eigenvalues
        )
        show_nmodes_matrix(eigensystem)
        show_nmodes_matrix(deuterated)
        for idx, mode in enumerate(deuterated_nmodes):
            mode_xyz = mode.to_MoleculeXYZ(
                comment=f'{idx}. Vibrational mode, {mode.frequency:.2f} cm-1,'
                ' deuterated',
            )
            print(mode_xyz)

    return deuterated_nmodes


def find_nmodes_displacement(
    start: Geometry,
    end: Geometry,
    nmodes: list[NormalMode],
) -> np.typing.NDArray[np.float64]:

    # Make sure that these are the same molecules
    for atom_s, atom_e in zip(start.atoms, end.atoms, strict=True):
        if atom_s.name != atom_e.name:
            raise ValueError(f"Mismatch between atoms {atom_s} and {atom_e}")

    dim = 3 * len(start.atoms)
    displacement_Descartes = np.zeros(shape=dim)
    for idx, (satom, eatom) in enumerate(
        zip(start.atoms, end.atoms, strict=True)
    ):
        displacement_Descartes[3 * idx:3 * idx + 3] = [
            e - s for e, s in zip(eatom.xyz, satom.xyz)
        ]

    mass_matrix = build_mass_matrix(start, masses_dict=ATOMIC_MASSES)
    displacement_mass_weighted = np.sqrt(mass_matrix) @ displacement_Descartes

    nmodes_matrix = build_nmodes_matrix(nmodes)
    nmodes_displacements = nmodes_matrix.T @ displacement_mass_weighted
    return nmodes_displacements


def main():
    xyz_path = "~/chemistry/cci/phenoxide/calculations/phenoxide/strontium/vib/dz/findiff/normal_modes.xyz"
    normal_modes = xyz_file_to_NormalModesList(xyz_path)
    # deuterate_modes(normal_modes, present_mode=True)


if __name__ == "__main__":
    main()
