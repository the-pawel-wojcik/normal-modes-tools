import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import normal_modes_tools as nmt
from normal_modes_tools.decomposition import find_nmodes_displacement
from normal_modes_tools.conversions import (
    c_SI, inv_cm_to_inv_m, aa_to_m, amu_to_kg, h_SI, inv_m_to_inv_cm,
)
from normal_modes_tools.huang_rhys_factors import huang_rhys_factor
import numpy as np
import prettytable
import xyz_parser as xyz


def huang_rhys_factor_test():
    mode_wavenumber_cm = 70.0

    print(f'ω = {mode_wavenumber_cm:.0f} cm-1')

    print('The first 5 energy levels of the harmonic oscillator:')
    levels = [mode_wavenumber_cm * (lvl  + 0.5) for lvl in range(5)]
    print('')
    print('\t'.join([f'{lvl:>4d}' for lvl in range(5)]))
    print('\t'.join([f'{lvl:4.0f}' for lvl in levels]))
    print('')

    print()
    table = prettytable.PrettyTable()
    table.field_names = [
        'Δq, Å√amu',
        'classic energy, cm-1',
        'Huang-Rhys factor',
    ]
    for dq in np.arange(0.0, 3.2, 0.2, dtype=float):
        energy_SI = 0.5 * (
            2 * np.pi * c_SI * mode_wavenumber_cm * inv_cm_to_inv_m
        )**2 * dq**2 * aa_to_m**2 * amu_to_kg 
        energy_inv_m = energy_SI / h_SI / c_SI
        energy_inv_cm = energy_inv_m * inv_m_to_inv_cm
        hrf = huang_rhys_factor(dq, mode_wavenumber_cm)
        table.add_row([dq, energy_inv_cm, hrf])
    table.float_format = '.1'
    table.custom_format['classic energy, cm-1'] = lambda _, v: f"{v:.0f}"
    table.align = 'r'
    table.set_style(prettytable.TableStyle.PLAIN_COLUMNS)
    print(table)


def huang_rhys_factor_plot(wavenumber: float = 70):

    height = 2.5
    figsize = (16/9 * height, height)
    fig = plt.figure(figsize=figsize, layout='constrained')
    ax = fig.subplots()

    xs = np.linspace(start=-3.0, stop=0.0, num=251)

    omega_SI = 2.0 * np.pi * c_SI * wavenumber * inv_cm_to_inv_m
    energies_SI = 0.5 * xs**2 * aa_to_m**2 * amu_to_kg * omega_SI **2
    energies_inv_cm = energies_SI / h_SI / c_SI * inv_m_to_inv_cm

    ax.plot(
        xs,
        energies_inv_cm,
        label='Harmonic potential\n'r'$\omega = $' f'{wavenumber:.0f}' 'cm$^{-1}$'
    )
    ax.yaxis.set_label_text("Energy, cm$^{-1}$")
    ax.xaxis.set_label_text(r'$\Delta q, \AA \sqrt{\text{amu}}$')
    for lvl in range(9):
        ax.hlines(
            y=wavenumber*(lvl + 0.5),
            xmin=0,
            xmax=3,
            ls=':',
            color='gray',
            lw=0.5,
        )

    ax2 = ax.twinx()
    xs2 = np.linspace(0.0, 3.0, num=251)
    hr_factors = huang_rhys_factor(xs2, wavenumber)  # typing: ignore
    ax2.plot(
        xs2,
        hr_factors,
        color='k',
        ls='dashed',
        lw=2,
        label='Huang-Rhys factors'
    )
    ax2.yaxis.set_major_locator(
        matplotlib.ticker.MultipleLocator(base=1.0, offset=0.0)
    )
    # ax2.grid(which='both', axis='both')
    ax2.set_ylabel('Huang-Rhys factor')

    ax.legend(loc='upper center')
    # ax2.legend()
    plt.show()
    fig.savefig("/home/pawel/chemistry/vibrations/notes/img/Huang-Rhys.pdf")


def find_nmodes_displacement_test():
    ground_nmds_path = "~/chemistry/cci/phenoxide/calculations/phenoxide/strontium/vib/dz/findiff/normal_modes.xyz"
    ground_nmds = nmt.xyz_file_to_NormalModesList(ground_nmds_path)

    geometries_path = "./displacements.xyz"
    with open(geometries_path, 'r', encoding='utf-8') as geometries_file:
        all_geometries = xyz.parse(geometries_file)

    initial_geometry = nmt.Geometry.from_MoleculeXYZ(all_geometries[0])
    for geo_idx, geometry_xyz in enumerate(all_geometries):
        target_geometry = nmt.Geometry.from_MoleculeXYZ(geometry_xyz)
        delta_nmodes = find_nmodes_displacement(
            start=initial_geometry,
            end=target_geometry,
            nmodes=ground_nmds,
        )

        dq_aa_sqrt_amu = delta_nmodes[0]
        print(
            f'Displacment along mode #0 in geometry #{geo_idx}: '
            f'{dq_aa_sqrt_amu:.2f}'
        )


def main():
    huang_rhys_factor_test()
    huang_rhys_factor_plot()


if __name__ == "__main__":
    main()
