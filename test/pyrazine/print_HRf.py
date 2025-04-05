from xyz_parser import dataclass
from normal_modes_tools.conversions import (
    c_SI, inv_cm_to_inv_m, aa_to_m, amu_to_kg, h_SI,
)
from normal_modes_tools.huang_rhys_factors import huang_rhys_factor
import numpy as np


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
