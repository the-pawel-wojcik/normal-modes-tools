from normal_modes_tools import NormalMode

irrep_to_order = {
    "C2v": {
        "a1": 0,
        "a2": 1,
        "b1": 2,
        "b2": 3,
    }, # From Herzberg's Book Table 13
}


def sort_Mulliken(
    mode: NormalMode,
    point_group: str = "C2v",
    irrep: str = "a1",
) -> tuple[int, float]:
    """ A key for sorting a list of `NormalMode` objects by the order suggested
    in Mulliken's convention [1]. Sort the modes first by the irrep, and then
    by frequency. Highest frequency goes first.

    [1] R. S. Mulliken, Report on Notation for the Spectra of Polyatomic
    Molecules, The Journal of Chemical Physics 23, 1997 (1955).
    """
    if mode.at.point_group != "":
        point_group = mode.at.point_group
    
    if mode.irrep != "":
        irrep = mode.irrep

    irrep_position = irrep_to_order[point_group][irrep.lower()]

    return irrep_position, -mode.frequency
