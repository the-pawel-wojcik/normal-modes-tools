import xyz_parser as xyz
import os
from dataclasses import dataclass
from atomic_masses import ATOMIC_MASSES
import numpy as np
from copy import deepcopy

deuterated_masses = deepcopy(ATOMIC_MASSES)
deuterated_masses['H'] = 2.0

xyz_fname = "~/chemistry/cci/phenoxide/calculations/phenoxide/strontium/vib/dz/findiff/normal_modes.xyz"
xyz_fname = os.path.expanduser(xyz_fname)

with open(xyz_fname) as xyz_file:
    molecules = xyz.parse(xyz_file)

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


nmodes_count = len(normal_modes)

# eigenvectors of the mass-weighted Hessian form columns
nmodes_matrix = np.zeros(
    shape=(nmodes_count + 6, nmodes_count),
    dtype=np.float128,
)

mass_matrix = np.zeros(
    shape=(nmodes_count + 6, nmodes_count + 6),
    dtype=np.float128,
)

deuterated_mass_matrix = np.zeros(
    shape=(nmodes_count + 6, nmodes_count + 6),
    dtype=np.float128,
)

freq_pow_2 = np.zeros(
    shape=(nmodes_count, nmodes_count),
    dtype=np.float128,
)

for column, mode in enumerate(normal_modes):
    freq_pow_2[column, column] = np.float128(mode.frequency) ** 2

    for atom_idx, atom in enumerate(mode.displacement):

        atom_mass = ATOMIC_MASSES[atom.name.capitalize()]
        mass_matrix[
            3*atom_idx:3*atom_idx+3, 3*atom_idx: 3*atom_idx+3
        ] = np.eye(N=3) * atom_mass 

        atom_deuterated_mass = deuterated_masses[atom.name.capitalize()]
        deuterated_mass_matrix[
            3*atom_idx:3*atom_idx+3, 3*atom_idx: 3*atom_idx+3
        ] = np.eye(N=3) * atom_deuterated_mass

        for cart_idx in range(3):
            row = atom_idx * 3 + cart_idx
            nmodes_matrix[row, column] = np.float128(atom.xyz[cart_idx])

weighted_D = np.sqrt(mass_matrix) @ nmodes_matrix 
hessian = weighted_D @ freq_pow_2 @ weighted_D.T

print(nmodes_matrix)
print(f'{mass_matrix=}')
print(f'{deuterated_mass_matrix=}')
print(f'{hessian=}')

import matplotlib.pyplot as plt

# plt.imshow(hessian)
# plt.imshow(np.log(deuterated_mass_matrix+1))
# plt.imshow(np.log(mass_matrix+1))
# plt.imshow(nmodes_matrix)
plt.imshow(weighted_D)
plt.colorbar()
plt.show()

