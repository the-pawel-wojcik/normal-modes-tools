import normal_modes_tools as nmt
import numpy as np
import json

gradient_json_filename = 'inputs/sroph_at_g0_kappa_a.json'
nmodes_xyz_fname = 'inputs/sroph_f0.xyz'
deuterated_nmodes_xyz_fname = 'outputs/sroph-5d_f0.xyz'


def print_np_vec_with_atoms(
    vec: np.typing.NDArray,
    atoms: list[nmt.AtomVector],
) -> None:
    natoms = len(atoms)
    assert (len(vec) == natoms*3)
    for atom_idx, atom in enumerate(atoms):
        print(
            f'{atom.name:<3}',
            ' '.join(
            f'{x:5.0f}' for x in vec[3*atom_idx: 3*atom_idx +3]
        ))


def print_gradient_in_normal_modes(
    gradient: np.typing.NDArray,
    nmodes: list[nmt.NormalMode],
) -> None:
    for mode, grad_comp in zip(nmodes, gradient):
        print(f'Gradient along mode of freq = {mode.frequency:4.0f}'
              f': {grad_comp:4.0f}')


def gradient_from_json(
    gradient_json_filename: str
) -> np.typing.NDArray:
    with open(gradient_json_filename, 'r', encoding='utf-8') as grad_json_file:
        grad_input = json.load(grad_json_file)

    dim = 33
    grad_input_np = np.zeros(shape=(dim), dtype=float)
    for idx, component in enumerate(grad_input['gradient']):
        grad_value = component['gradient, cm-1']
        grad_input_np[idx] = grad_value

    return grad_input_np


def main():
    nmodes = nmt.xyz_file_to_NormalModesList(nmodes_xyz_fname)
    ref_geo = nmodes[0].at

    grad_input_np = gradient_from_json(gradient_json_filename)
    print_gradient_in_normal_modes(grad_input_np, nmodes)

    mass_matrix = nmt.build_mass_matrix(ref_geo, nmt.ATOMIC_MASSES)
    nmodes_matrix = nmt.build_nmodes_matrix(nmodes)
    mass_matrix_sqrt = np.sqrt(mass_matrix)
    grad_descartes = mass_matrix_sqrt @ nmodes_matrix @ grad_input_np

    print("Gradient in Cartesian coordinates")
    print_np_vec_with_atoms(grad_descartes, ref_geo.atoms)

    deuterated_nmodes = nmt.xyz_file_to_NormalModesList(
        deuterated_nmodes_xyz_fname
    )
    deuterated_nmodes_matrix = nmt.build_nmodes_matrix(deuterated_nmodes)
    deuterated_mm = nmt.build_mass_matrix(ref_geo, nmt.DEUTERATED_MASSES)
    deuterated_mm_inv_sqrt = nmt.get_mass_inv_sqrt(deuterated_mm)
    grad_deuterated_nmodes = \
        deuterated_nmodes_matrix.T @ deuterated_mm_inv_sqrt @ grad_descartes

    print_gradient_in_normal_modes(grad_deuterated_nmodes, deuterated_nmodes)

    
if __name__ == "__main__":
    main()
