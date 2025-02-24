from __future__ import annotations
import xyz_parser as xyz
from dataclasses import dataclass
from normal_modes_tool.atomic_masses import ATOMIC_MASSES
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


@dataclass
class NormalMode:
    frequency: float
    displacement: list[AtomVector]
    at: Geometry

    def __str__(self) -> str:
        fmt = '-13.8f'
        str_xyz = f"{len(self.displacement)}\n"
        str_xyz += f"Comment\n"
        for geo, nmd in zip(self.at.atoms, self.displacement):
            assert geo.name == nmd.name
            str_xyz += f"{geo.name:<3}"
            for coord in geo.xyz:
                str_xyz += f"{coord:{fmt}}"
            for coord in nmd.xyz:
                str_xyz += f"{coord:{fmt}}"
            str_xyz += '\n'
        str_xyz = str_xyz[:-1]  # trim trailin new line

        return str_xyz

    def get_numpy(self) -> np.typing.NDArray[np.float64]:
        array = np.zeros(shape=3 * len(self.displacement), dtype=np.float64)
        for idx, atom in enumerate(self.displacement):
            array[3 * idx: 3 * idx + 3] = deepcopy(atom.xyz)
        return array
    
    def __matmul__(self, other: NormalMode) -> float:
        if not isinstance(other, NormalMode):
            return NotImplemented
        
        left = self.get_numpy()
        left /= np.linalg.norm(left)

        right = other.get_numpy()
        right /= np.linalg.norm(right)

        inner = np.dot(left, right)
        return inner

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


def moleculeXYZ_to_NormalModesList(
    xyz_nmodes: list[xyz.MoleculeXYZ]
) -> list[NormalMode]:
    normal_modes: list[NormalMode] = list()
    for molecule in xyz_nmodes:
        normal_modes.append(NormalMode.from_MoleculeXYZ(molecule))

    return normal_modes


def collect_normal_modes(
    xyz_path: str,
) -> list[NormalMode]:
    nmodes_xyz = xyz.read_xyz_file(xyz_path)
    nmodes = moleculeXYZ_to_NormalModesList(nmodes_xyz)
    return nmodes


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
) -> np.linalg._linalg.EigResult:

    mass_inv_sqrt = get_mass_inv_sqrt(mass_matrix)
    mass_weighted_hessian = mass_inv_sqrt @ hessian @ mass_inv_sqrt
    eigensystem = linalg.eig(mass_weighted_hessian)
    return eigensystem


def print_eigenvalues(
    evals: np.typing.NDArray[np.float64],
    show_tr_rot: bool = False,
) -> None:
    """ `evals` is expected to be the value of the `eigenvalues` attribut of
    the return value of np.linalg.eig funciton. """
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
    `eigenvalues` attribut of the return value of np.linalg.eig funciton. """
    left_freqs = [np.sqrt(freq.real) for freq in left if freq.real > 1]
    right_freqs = [np.sqrt(freq.real) for freq in right if freq.real > 1]

    if len(left_freqs) != len(right_freqs):
        raise RuntimeError("Mismatch in the number of frequencies.")
    
    for idx, (left, right) in enumerate(zip(sorted(left_freqs), sorted(right_freqs))):
        print(f'{idx + 1:3d}: {left:>4.0f} {right:>4.0f}')

def str_eigenvalue(eval):
    if eval.real > 0:
        return f"{np.sqrt(eval.real):4.0f}"
    else:
        return f"{0}"


def show_nmodes_matrix(eigensystem: np.linalg._linalg.EigResult) -> None:
    _, ax = plt.subplots()
    nmodes = np.matrix(eigensystem.eigenvectors)
    freqs = [str_eigenvalue(eval) for eval in eigensystem.eigenvalues]
    ax.imshow(nmodes.real, aspect='equal')
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
    displaced = Geometry.from_numpy(displacement_Descartes, atom_names=atom_names)
    return displaced


def main():
    xyz_path = "~/chemistry/cci/phenoxide/calculations/phenoxide/strontium/vib/dz/findiff/normal_modes.xyz"
    normal_modes = collect_normal_modes(xyz_path)

    equilibrium_Descarte = normal_modes[0].at
    mass_matrix = build_mass_matrix(equilibrium_Descarte, ATOMIC_MASSES)
    hessian = build_hessian(normal_modes, mass_matrix)
    eigensystem = diagonalize_hessian(hessian, mass_matrix) 

    deuterated_masses = deepcopy(ATOMIC_MASSES)
    deuterated_masses['H'] = 2.0
    deuterated_mm = build_mass_matrix(equilibrium_Descarte, deuterated_masses)
    deuterated = diagonalize_hessian(hessian, deuterated_mm)

    # print_pair_of_eigenvalues(eigensystem.eigenvalues, deuterated.eigenvalues)
    # show_nmodes_matrix(eigensystem)
    # show_nmodes_matrix(deuterated)

    # These are normalized
    # plt.imshow(nmodes_matrix.T @ nmodes_matrix)
    # plt.show()
    for displacement in [0.00, 0.01, 0.02, 0.03, 0.04]:
        displaced_mode = 0
        displaced = generated_displaced_geometry(
            which_mode=displaced_mode,
            displacement=displacement,
            reference_geometry=equilibrium_Descarte,
            normal_modes=normal_modes,
            mass_matrix=mass_matrix,
        )
        displaced_xyz = xyz.MoleculeXYZ.from_Geometry(
            displaced, comment = f'Displaced {displacement} Å√(amu) along'
            f' mode {displaced_mode}',
        )
        print(displaced_xyz)


if __name__ == "__main__":
    main()
