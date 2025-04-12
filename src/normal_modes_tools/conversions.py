import numpy as np

# Mass
amu_to_kg = 1.66053906892e-27
amu_to_au = 1822.8884843

# Distance
aa_to_m = 1e-10
aa_to_au = 1.889716

# Energy
inv_cm_to_inv_m = 100
inv_m_to_inv_cm = 0.01
Eh_to_eV = 27.211386
Eh_to_cm = 219474.6301460127
eV_to_cm = 8065.543965530189
eV_to_Eh = 0.0367493225078649
cm_to_Eh = 4.556335278180979e-06
cm_to_eV = 0.000123984198

# constants
c_SI = 2.99792458e8
h_SI = 6.62607015e-34

c_au = 137.0359996


def dq_to_dQ(dq: float, wavenumber: float) -> float:
    """Transform a displacment along a single mode from normal coordinates to
    dimensionless normal coordinates.

    `dq` is expected in the units of Å * √amu.
    `wavenumber` is expected in the units of inverse centimeters.
    """
    omega_SI = 2.0 * np.pi * c_SI * wavenumber * inv_cm_to_inv_m
    shift_DNC = (
        np.sqrt(2.0 * np.pi * omega_SI / h_SI)
        * 
        dq * aa_to_m * amu_to_kg**0.5
    )
    return shift_DNC
