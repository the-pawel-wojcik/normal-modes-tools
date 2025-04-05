from typing import overload
import numpy as np
from normal_modes_tools.conversions import (
    c_SI, inv_cm_to_inv_m, aa_to_m, amu_to_kg, h_SI,
)

@overload
def huang_rhys_factor(
    dq_aa_sqrt_amu: float,
    mode_wavenumber_cm: float,
) -> float: ...


@overload
def huang_rhys_factor(
    dq_aa_sqrt_amu: np.typing.NDArray,
    mode_wavenumber_cm: float,
) -> np.typing.NDArray: ...


def huang_rhys_factor(
    dq_aa_sqrt_amu: float | np.typing.NDArray,
    mode_wavenumber_cm: float,
) -> float | np.typing.NDArray:
    """ Find the Huang-Rhys factors for a displacement of `dq_aa_sqrt_amu`
    along normal with a harmonic wavenumber `mode_wavenumber_cm`. 

    Huang-Rhys factors arise from a comparison of the energy of a classical and
    quantum harmonic oscillators. The energy of a classical harmonic oscillator
    is given by its displacement, `dq_aa_sqrt_amu`. Plugging this energy to the
    formula for the energy of a quantum harmonic oscillator does not make sense
    as it typically does not correspond to any integer quantum number. This
    non-integer "effective" quantum number is Huang-Rhys factor.

    In a parallel approximation the Huang-Rhys factor gives an estimate for the
    quantum numbers that will be close to the maximum of a vibrational
    progression.

    The input displacement is allowed to be a list too because Paweł needed it
    to make a pretty plot.
    """
    hrf = 0.5 * (
        (2 * np.pi)**2 * c_SI * mode_wavenumber_cm * inv_cm_to_inv_m 
        * dq_aa_sqrt_amu**2 * aa_to_m**2  * amu_to_kg
        / h_SI
        - 1
    )
    return hrf
