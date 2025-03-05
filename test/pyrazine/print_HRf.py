from xyz_parser import dataclass
import numpy as np

amu_to_kg = 1.66053906892e-27
aa_to_m = 1e-10
c_SI = 2.99792458e8
inv_cm_to_inv_m = 100
h_SI = 6.62607015e-34

inv_m_to_inv_cm = 0.01

def huang_rhys_factor(
    dq_aa_sqrt_amu: float,
    mode_wavenumber_cm: float,
) -> float:
    hrf = 0.5 * (
        (2 * np.pi)**2 * c_SI * mode_wavenumber_cm * inv_cm_to_inv_m 
        * dq_aa_sqrt_amu**2 * aa_to_m**2  * amu_to_kg
        / h_SI
        - 1
    )
    return hrf

@dataclass
class Displacement:
    name: str
    dq: float
    wavenumber: float


def dq_to_dQ(dq: float, wavenumber: float) -> float:
    """Transform a displacment along a single mode from normal coordinates to
    dimensionless normal coordinates."""
    omega_SI = 2.0 * np.pi * c_SI * wavenumber * inv_cm_to_inv_m
    shift_DNC = (
        np.sqrt(2.0 * np.pi * omega_SI / h_SI)
        * 
        dq * aa_to_m * amu_to_kg**0.5
    )
    return shift_DNC

def main():
    geometry_shifts = [
        Displacement(name='5', dq=-0.251, wavenumber=618.13),
        Displacement(name='4', dq=0.039, wavenumber=1069.15),
        Displacement(name='3', dq=0.151, wavenumber=1274.83),
        Displacement(name='2', dq=-0.042, wavenumber=1675.86),
        Displacement(name='1', dq=-0.008, wavenumber=3245.44),
    ]
    for shift in geometry_shifts:
        dQ = dq_to_dQ(shift.dq, shift.wavenumber)
        print(f'Shift dq={shift.dq:6.3f} Å√amu, or in DNC dQ = {dQ:5.2f},'
              f' along mode #{shift.name}'
              f' ({shift.wavenumber:4.0f}) produces RH factor = '
              f'{huang_rhys_factor(shift.dq, shift.wavenumber):.2f}')


    
if __name__ == "__main__":
    main()
