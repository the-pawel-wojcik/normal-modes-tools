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
        print(f'{idx:3d}({mode.irrep.lower()}): {mode.frequency:>4.0f}')
    
