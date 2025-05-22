from normal_modes_tools.geometry import Geometry
from normal_modes_tools import NormalMode, build_nmodes_matrix
from normal_modes_tools.atomic_masses import ATOMIC_MASSES
import numpy as np
from numpy.typing import NDArray


def find_nmodes_displacement(
    start: Geometry,
    end: Geometry,
    nmodes: list[NormalMode],
) -> NDArray[np.float64]:
    """ 
    Displacement along normal modes:
        dq = D^T M ^{1/2} (end - start)
    The units of displacement are L * sqrt(amu), amu stands for atomic mass
    units. L stands for the units of the geometries.
    D is a matrix that stores the normal modes as its columns.
    M is a matrix of atomic masses.
    """

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

    mass_matrix = start.get_mass_matrix(masses_dict=ATOMIC_MASSES)
    displacement_mass_weighted = np.sqrt(mass_matrix) @ displacement_Descartes

    nmodes_matrix = build_nmodes_matrix(nmodes)
    nmodes_displacements = nmodes_matrix.T @ displacement_mass_weighted
    return nmodes_displacements


