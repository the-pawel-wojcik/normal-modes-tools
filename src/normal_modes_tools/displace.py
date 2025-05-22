import numpy as np
from numpy.typing import NDArray
from normal_modes_tools.geometry import Geometry
from normal_modes_tools.normal_mode import NormalMode, build_nmodes_matrix
from normal_modes_tools.atomic_masses import ATOMIC_MASSES
from normal_modes_tools.util import get_mass_inv_sqrt


def generate_displaced_geometry(
    which_mode: int,
    displacement: float,
    reference_geometry: Geometry,
    normal_modes: list[NormalMode],
    mass_matrix: NDArray[np.float64],
) -> Geometry:
    """ Build a new molecular geometry by displacing the reference geometry by
    `displacement` AA sqrt(amu) along the normal mode number `which_mode` """
    len_nmodes = len(normal_modes)

    if not which_mode in range(len_nmodes):
        raise ValueError(
            f'The mode number must be in the range [0, {len_nmodes}).'
        )
    
    nmodes_matrix = build_nmodes_matrix(normal_modes)

    displaced_vector_normal_coordinates = np.zeros(shape=(len_nmodes))
    displaced_vector_normal_coordinates[which_mode] = displacement
    displaced_vector_mass_weighted =\
            nmodes_matrix @ displaced_vector_normal_coordinates
    mass_inv_sqrt = get_mass_inv_sqrt(mass_matrix)
    displacement_Descartes = (
        mass_inv_sqrt @ displaced_vector_mass_weighted
        + 
        reference_geometry.to_numpy()
    )

    atom_names = [atom.name for atom in reference_geometry.atoms]
    displaced = Geometry.from_numpy(
        vec=displacement_Descartes,
        atom_names=atom_names,
    )
    return displaced


def displace_main(
    nmodes: list[NormalMode],
    mode_idx: int,
    dq: float,
) -> None:
    ref_geo: Geometry = nmodes[0].at
    mass_matrix = ref_geo.get_mass_matrix(ATOMIC_MASSES)
    new_geo = generate_displaced_geometry(
        which_mode=mode_idx,
        displacement=dq,
        reference_geometry=ref_geo,
        normal_modes=nmodes,
        mass_matrix=mass_matrix,
    )
    print(ref_geo.xyz_str('Reference Geometry'))
    print(new_geo.xyz_str(f'Geometry displaced by {dq} along mode {mode_idx}'))
