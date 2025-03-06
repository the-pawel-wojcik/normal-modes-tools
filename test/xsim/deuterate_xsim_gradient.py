import normal_modes_tools as nmt
import numpy as np
import json
import argparse


gradient_json_filename = 'inputs/sroph_at_g0_kappa_a.json'
nmodes_xyz_fname = 'inputs/sroph_f0.xyz'
deuterated_nmodes_xyz_fname = 'outputs/sroph-5d_f0.xyz'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gradient_fname',
        default=gradient_json_filename,
    )
    parser.add_argument(
        '--debug',
        help='Print intermediates',
        default=False,
        action='store_true',
    )
    args = parser.parse_args()
    return args


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


def collect_gradient_from_json(
    gradient_json_filename: str,
) -> dict:
    with open(gradient_json_filename, 'r', encoding='utf-8') as grad_json_file:
        grad_input = json.load(grad_json_file)
    return  grad_input


def gradient_json_to_numpy(
    gradient_json: dict,
) -> np.typing.NDArray:

    dim = len(gradient_json['gradient'])

    grad_input_np = np.zeros(shape=(dim), dtype=float)
    for idx, component in enumerate(gradient_json['gradient']):
        grad_value = component['gradient, cm-1']
        grad_input_np[idx] = grad_value

    return grad_input_np


def main():
    args = get_args()
    DEBUG = args.debug
    nmodes = nmt.xyz_file_to_NormalModesList(nmodes_xyz_fname)
    ref_geo = nmodes[0].at

    gradient_json = collect_gradient_from_json(args.gradient_fname)
    grad_input_np = gradient_json_to_numpy(gradient_json)
    if DEBUG:
        print_gradient_in_normal_modes(grad_input_np, nmodes)

    mass_matrix = nmt.build_mass_matrix(ref_geo, nmt.ATOMIC_MASSES)
    nmodes_matrix = nmt.build_nmodes_matrix(nmodes)
    mass_matrix_sqrt = np.sqrt(mass_matrix)
    grad_descartes = mass_matrix_sqrt @ nmodes_matrix @ grad_input_np

    if DEBUG:
        print("Gradient in Cartesian coordinates")
        print_np_vec_with_atoms(grad_descartes, ref_geo.atoms)

    deuterated_nmodes = nmt.xyz_file_to_NormalModesList(
        deuterated_nmodes_xyz_fname
    )
    deuterated_nmodes_matrix = nmt.build_nmodes_matrix(deuterated_nmodes)
    deuterated_mm = nmt.build_mass_matrix(ref_geo, nmt.DEUTERATED_MASSES)
    deuterated_mm_inv_sqrt = nmt.get_mass_inv_sqrt(deuterated_mm)
    deuterated_gradient = \
        deuterated_nmodes_matrix.T @ deuterated_mm_inv_sqrt @ grad_descartes

    if DEBUG:
        print_gradient_in_normal_modes(
            gradient=deuterated_gradient,
            nmodes=deuterated_nmodes,
        )

    outpack = {
        'gradient': list(),
        'EOM states': gradient_json['EOM states'],
    }
    for idx, (mode, grad_comp) in enumerate(
        zip(deuterated_nmodes, deuterated_gradient)
    ):
        outpack['gradient'].append({
            'mode #': idx,
            'frequency, cm-1': mode.frequency,
            'gradient, cm-1': grad_comp,
        })

    print(json.dumps(outpack))

    
if __name__ == "__main__":
    main()
