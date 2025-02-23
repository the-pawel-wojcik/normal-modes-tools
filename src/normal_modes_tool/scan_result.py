import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


Ha_to_J = 4.3597447222060e-18
amu_to_kg = 1.66053906892e-27
aa_to_m = 1e-10
c_SI = 2.99792458e8
inv_m_to_inv_cm = 0.01


def produce_omega():
    pes = [
        {
            'energy, Ha': -3402.995135818499875,
            'displacement': 0.00,
        },
        {
            'energy, Ha': -3402.995135790143650,
            'displacement': 0.01,
        },
        {
            'energy, Ha': -3402.995135705070425,
            'displacement': 0.02,
        },
        {
            'energy, Ha': -3402.995135563234726,
            'displacement': 0.03,
        },
        {
            'energy, Ha': -3402.995135364740236,
            'displacement': 0.04,
        },
    ]
    energies = [point['energy, Ha'] for point in pes]
    ab_initio_minimum = min(energies)
    shifted_energies = [e - ab_initio_minimum for e in energies]

    displacements = [point['displacement'] for point in pes]

    def parabola(x, omega_square, linear, energy_minimum):
        x = np.array(x)
        return 0.5 * omega_square * x ** 2  + linear * x + energy_minimum

    parameters, covariance = curve_fit(
        f=parabola,
        xdata=displacements,
        ydata=shifted_energies,
    )

    fitted_omega_square = parameters[0]
    fitted_linear = parameters[1]
    fitted_energy_minimum = parameters[2]
    print(f'{fitted_omega_square=}')
    print(f'{fitted_linear=}')
    print(f'{fitted_energy_minimum=}')

    fit_energies = parabola(displacements, *parameters)
    plt.plot(displacements, shifted_energies, 'o', label='ab initio')
    plt.plot(displacements, fit_energies, '-', label='fit')
    plt.legend()
    plt.show()


def main():
    # The dimension of the fitted lambda is 1/T**2
    # The unit of the fitted lambda is Ha / (amu * AA**2).
    fitted_lambda = 5.672e-4

    omega_square_SI = fitted_lambda * Ha_to_J / amu_to_kg / aa_to_m / aa_to_m
    omega_SI = np.sqrt(omega_square_SI)
    omega_inv_m = omega_SI / 2.0 / np.pi / c_SI
    omega_inv_cm = omega_inv_m * inv_m_to_inv_cm
    print(f'{omega_inv_cm=}')

if __name__ == "__main__":
    produce_omega()
    main()
