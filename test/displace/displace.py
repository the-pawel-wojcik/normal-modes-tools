import normal_modes_tools as nmt
import xyz_parser as xyz
import numpy as np

def displace_along_mode(
    normal_modes: list[nmt.NormalMode],
    equilibrium_Descarte: nmt.Geometry,
    mass_matrix: np.typing.NDArray[np.float64],
    displaced_mode: int = 0,
) -> None:
    for displacement in [0.00, 0.01, 0.02, 0.03, 0.04]:
        displaced = nmt.generated_displaced_geometry(
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


def main():
    xyz_path = "inputs/SrOPh_f0.xyz"
    normal_modes = nmt.xyz_file_to_NormalModesList(xyz_path)
    mass_matrix = nmt.build_mass_matrix(normal_modes[0].at, nmt.ATOMIC_MASSES)

    displace_along_mode(
        normal_modes=normal_modes,
        equilibrium_Descarte=normal_modes[0].at,
        mass_matrix=mass_matrix,
        displaced_mode=10,
    )

if __name__ == "__main__":
    main()
