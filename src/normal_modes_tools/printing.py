from typing import Any

from normal_modes_tools.normal_mode import NormalMode

def pretty_print(
    nmodes: list[NormalMode],
    latex: bool = False,
    sort_key: Any = None,
) -> None:
    if sort_key is not None:
        nmodes.sort(key=sort_key)

    for idx, mode in enumerate(nmodes, start=1):
        irrep = f'{mode.irrep.lower()}'
        if latex:
            irrep = '$' + irrep[0] + '_{' + irrep[1:] + '}$'
            print(f'{idx:3d} & {irrep:8} & {mode.frequency:>4.0f} \\\\')
        else:
            irrep = f'({irrep})'
            print(f'{idx:3d}{irrep:5}: {mode.frequency:>4.0f}')
    
