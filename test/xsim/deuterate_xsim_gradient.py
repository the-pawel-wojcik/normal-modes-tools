import normal_modes_tools as nmt
import numpy as np
import json

nmodes_xyz_filename = './inputs/sroph_f0.xyz'
gradient_json_filename = 'inputs/sroph_at_g0_kappa_a.json'


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


def main():
    nmodes = nmt.xyz_file_to_NormalModesList(nmodes_xyz_filename)
    deuterated_nmodes = nmt.deuterate_modes(nmodes)

    ref_geo = nmodes[0].at
    mass_matrix = nmt.build_mass_matrix(ref_geo, nmt.ATOMIC_MASSES)
    nmodes_matrix = nmt.build_nmodes_matrix(nmodes)


    with open(gradient_json_filename, 'r', encoding='utf-8') as grad_json_file:
        grad_input = json.load(grad_json_file)

    dim = 33
    grad_input_np = np.zeros(shape=(dim), dtype=float)
    for idx, component in enumerate(grad_input['gradient']):
        frequency = component['frequency, cm-1']
        grad_value = component['gradient, cm-1']
        grad_input_np[idx] = grad_value

    with np.printoptions(precision=0, suppress=True):
        print(grad_input_np)

    mass_matrix_sqrt = np.sqrt(mass_matrix)
    grad_descartes = mass_matrix_sqrt @ nmodes_matrix @ grad_input_np

    print("Gradient in Cartesian coordinates")
    print_np_vec_with_atoms(grad_descartes, ref_geo.atoms)
    
if __name__ == "__main__":
    main()
