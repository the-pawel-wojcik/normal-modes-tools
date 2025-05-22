import argparse
import numpy as np
from numpy.typing import NDArray
from argparse import ArgumentError, Action


def get_mass_inv_sqrt(
    mass_matrix: NDArray[np.float64]
) -> NDArray[np.float64]:

    mass_sqrt = np.sqrt(mass_matrix)
    mass_inv_sqrt = np.zeros(
        shape=mass_sqrt.shape,
        dtype=mass_sqrt.dtype
    )
    diagonal = mass_sqrt.diagonal()
    np.fill_diagonal(mass_inv_sqrt, 1.0 / diagonal)

    return mass_inv_sqrt



class DisplaceType(Action):
    """ This class enables specification of the mode displacement. """

    def __call__(
        self,
        parser,
        namespace,
        values,
        option_string=None,
    ):
        setattr(namespace, self.dest, dict())

        usage = '\nUse: --displace "mode_idx=INT dq=FLOAT"'

        for value in values.split():
            try:
                key, number_str = value.split('=')
            except Exception as e:
                raise RuntimeError(usage) from e
            if key == 'mode_idx':
                try:
                    mode_idx = int(number_str)
                except ValueError as e:
                    raise ArgumentError(
                        self,
                        'Key to `mode_idx` must be an int.' + usage,
                    ) from e
                getattr(namespace, self.dest)[key] = mode_idx
            elif key == 'dq':
                try:
                    dq = float(number_str)
                except ValueError as e:
                    raise ArgumentError(
                        self,
                        'Key to `dq` must be a float.' + usage,
                    ) from e
                getattr(namespace, self.dest)[key] = dq
            else:
                raise ArgumentError(
                    self,
                    message=f'Unrecoginized key {key}.' + usage,
                )

        if len(getattr(namespace, self.dest)) != 2:
            raise argparse.ArgumentError(
                argument=self,
                message=f'Incorrect number of arguments.' + usage,
            )
