import argparse
import numpy as np
from numpy import linalg
from copy import deepcopy
import matplotlib.pyplot as plt
from .geometry import Geometry
from .normal_mode import NormalMode, xyz_file_to_NormalModesList
from .atomic_masses import *
from . import mulliken
from .compare import show_duszynski
from .printing import pretty_print


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
    mass_matrix = equilibrium_Descarte.get_mass_matrix(ATOMIC_MASSES)
    hessian = build_hessian(normal_modes, mass_matrix)
    eigensystem = diagonalize_hessian(hessian, mass_matrix) 

    deuterated_mm = equilibrium_Descarte.get_mass_matrix(DEUTERATED_MASSES)
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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('xyz_file', help='File with normal modes')
    parser.add_argument(
        '--compare',
        help='Pass second set of normal modes. Compare then by drawing a '
        'Duszyński matrix (matrix of normal mode\'s inner products).',
        type=str,
    )
    parser.add_argument(
        '--deuterate',
        help='Print deuterated normal modes',
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--latex',
        help='Use latex when printing output.',
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--list',
        help='Print normal modes with their frequencies.',
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--Mulliken',
        help='Print normal modes using the Mulliken\'s convention.',
        default=False,
        action='store_true',
    )
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    xyz_path = args.xyz_file
    normal_modes = xyz_file_to_NormalModesList(xyz_path)

    if args.compare is not None:
        second_normal_modes = xyz_file_to_NormalModesList(args.compare)
        show_duszynski(normal_modes, second_normal_modes)

    if args.deuterate is True:
        deuterate_modes(normal_modes, present_mode=True)

    if args.list is True:
        latex = args.latex
        pretty_print(normal_modes, latex=latex)

    if args.Mulliken is True:
        pretty_print(normal_modes, sort_key=mulliken.sort_Mulliken)
        
        
if __name__ == "__main__":
    main()
