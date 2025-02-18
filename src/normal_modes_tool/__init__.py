from numpy.typing import NDArray
import xyz_parser as xyz
from dataclasses import dataclass
from atomic_masses import ATOMIC_MASSES
import numpy as np
from numpy import geomspace, linalg
from copy import deepcopy
from pathlib import Path
import os
import matplotlib.pyplot as plt

@dataclass
class AtomVector:
    name: str
    xyz: list[float]

@dataclass
class Geometry:
    atoms: list[AtomVector]

@dataclass
class NormalMode:
    frequency: float
    displacement: list[AtomVector]
    at: Geometry



# plt.imshow(hessian)
# deuterated_hessian = deut_mass_inv_sqrt @ hessian @ deut_mass_inv_sqrt
# plt.imshow(deuterated_hessian)
# for freq in deuterated_evals:
#     print(f'{freq=}')
#

# print(nmodes_matrix)
# print(f'{mass_matrix=}')
# print(f'{deuterated_mass_matrix=}')
# print(f'{hessian=}')


# plt.imshow(hessian)
# plt.imshow(np.log(deuterated_mass_matrix+1))
# plt.imshow(np.log(mass_matrix+1))
# plt.imshow(nmodes_matrix)
# plt.imshow(weighted_D)
# plt.colorbar()
# plt.show()


def collect_normal_modes() -> list[NormalMode]:
    xyz_file_path = os.path.expanduser("~/chemistry/cci/phenoxide/calculations/phenoxide/strontium/vib/dz/findiff/normal_modes.xyz"
    )
    with open(xyz_file_path) as xyz_file:
        molecules = xyz.parse(xyz_file)


    normal_modes: list[NormalMode] = list()
    for molecule in molecules:
        frequency = float(molecule.comment.split()[3])
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
            normalmode.displacement.append(AtomVector(
                name = atom.symbol,
                xyz = atom.extra,
            ))

        normal_modes.append(normalmode)

    return normal_modes


def build_mass_matrix(
    geometry: Geometry,
    masses_dict: dict[str, float]
) -> np.typing.NDArray[np.float64]:
    dim = 3 * len(geometry.atoms)
    diagonal = np.zeros(shape=dim, dtype=np.float64)
    for idx, atom in enumerate(geometry.atoms):
        atom_mass = np.float64(masses_dict[atom.name])
        diagonal[3 * idx: 3 * idx + 3] = [atom_mass for _ in range(3)]
    mass_matrix = np.zeros(shape=(dim, dim), dtype=np.float64)
    np.fill_diagonal(mass_matrix, diagonal)
    return mass_matrix


def build_hessian(
    normal_modes: list[NormalMode],
    mass_matrix: np.typing.NDArray[np.float64]
) -> np.typing.NDArray[np.float64]:

    len_nm = len(normal_modes)
    dim = 3 * len(normal_modes[0].at.atoms)

    # eigenvectors of the mass-weighted Hessian form columns
    nmodes_matrix = np.zeros(
        shape=(dim, len_nm),
        dtype=np.float64,
    )

    freq_matrix = np.zeros(
        shape=(len_nm, len_nm),
        dtype=np.float64,
    )

    for column, mode in enumerate(normal_modes):
        freq_matrix[column, column] = np.float64(mode.frequency)
        for atom_idx, atom in enumerate(mode.displacement):
            for cart_idx in range(3):
                row = atom_idx * 3 + cart_idx
                nmodes_matrix[row, column] = np.float64(atom.xyz[cart_idx])

    weighted_D = np.sqrt(mass_matrix) @ nmodes_matrix 
    hessian = weighted_D @ freq_matrix**2 @ weighted_D.transpose()

    return hessian


def get_mass_inv_sqrt(
        mass_sqrt: np.typing.NDArray[np.float64]
) -> np.typing.NDArray[np.float64]:

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

    mass_sqrt = np.sqrt(mass_matrix)
    mass_inv_sqrt = get_mass_inv_sqrt(mass_sqrt)
    mass_weighted_hessian = mass_inv_sqrt @ hessian @ mass_inv_sqrt
    eigensystem = linalg.eig(mass_weighted_hessian)
    evals = eigensystem.eigenvalues
    evects = eigensystem.eigenvectors
    freqs = [np.sqrt(freq.real) for freq in evals if freq.real > 1]
    
    for idx, freq in enumerate(sorted(freqs)):
        print(f'{idx:3d}: {freq:>4.0f}')
    return eigensystem



def main():
    normal_modes = collect_normal_modes()
    mass_matrix = build_mass_matrix(normal_modes[0].at, ATOMIC_MASSES)
    hessian = build_hessian(normal_modes, mass_matrix)
    eigensystem = diagonalize_hessian(hessian, mass_matrix) 


    # deuterated_masses = deepcopy(ATOMIC_MASSES)
    # deuterated_masses['H'] = 2.0
    # deuterated_mass_matrix = build_mass_matrix(normal_modes[0].at, ATOMIC_MASSES)



if __name__ == "__main__":
    main()
