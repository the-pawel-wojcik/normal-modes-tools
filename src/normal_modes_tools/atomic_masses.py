""" Data storage module with one helper function. """
def normalize_symbol(original: str) -> str:
    return original[0].upper() + original[1:].lower()


ATOMIC_MASSES = {
    "H":    1.00783,
    "He":   4.00260,
    "Li":   7.01600,
    "Be":   9.01218,
    "B":   11.00931,
    "C":   12.00000,
    "N":   14.00307,
    "O":   15.99491,
    "F":   18.99840,
    "Ne":  19.99244,
    "Na":  22.98980,
    "Mg":  23.98504,
    "Al":  26.98153,
    "Si":  27.97693,
    "P":   30.97376,
    "S":   31.97207,
    "Cl":  34.96885,
    "Ar":  39.948,
    "K":   38.96371,
    "Ca":  39.96259,
    "Sc":  44.95592,
    "Ti":  47.90,
    "V":   50.9440,
    "Cr":  51.9405,
    "Mn":  54.9380,
    "Fe":  55.9349,
    "Co":  58.9332,
    "Ni":  57.9353,
    "Cu":  62.9296,
    "Zn":  63.9291,
    "Ga":  68.9257,
    "Ge":  73.9219,
    "As":  74.9216,
    "Se":  79.9165,
    "Br":  78.9183,
    "Kr":  83.80,
    "Rb":  84.9117,
    "Sr":  87.9056,
    "Y":   88.9059,
    "Zr":  89.9043,
    "Nb":  92.9060,
    "Mo":  97.9055,
    "Tc":  98.9062,
    "Ru": 101.9037,
    "Rh": 102.9048,
    "Pd": 105.9032,
    "Ag": 106.90509,
    "Cd": 113.9036,
    "In": 114.9041,
    "Sn": 118.69,
    "Sb": 120.9038,
    "Te": 129.9067,
    "I":  126.9044,
    "Xe": 131.9042,
    "Cs": 132.9051,
    "Ba": 137.9050,
    "La": 138.9061,
    "Ce": 139.9053,
    "Pr": 140.9074,
    "Nd": 141.9075,
    "Pm": 144.913,
    "Sm": 151.9195,
    "Eu": 152.9209,
    "Gd": 157.9241,
    "Tb": 159.9250,
    "Dy": 163.9288,
    "Ho": 164.9303,
    "Er": 165.9304,
    "Tm": 168.9344,
    "Yb": 173.9390,
    "Lu": 174.9409,
    "Hf": 179.9468,
    "Ta": 180.9480,
    "W":  183.9510,
    "Re": 186.9560,
    "Os": 192.0,
    "Ir": 192.9633,
    "Pt": 194.9648,
    "Au": 196.9666,
    "Hg": 201.9706,
    "Tl": 204.9745,
    "Pb": 207.9766,
    "Bi": 208.9804,
    "Po": 208.9825,
    "At": 209.987,
    "Rn": 222.0175,
    "Fr": 223.0198,
    "Ra": 226.0254,
    "Ac": 227.0278,
    "Th": 232.0382,
    "Pa": 231.0359,
    "U":  238.0508,
    "Np": 237.0480,
    "Pu": 244.064,
    "Am": 243.0614,
    "Cm": 247.070,
    "Bk": 247.0702,
    "Cf": 251.080,
    "Es": 254.0881,
    "Fm": 257.095,
}

DEUTERATED_MASSES = ATOMIC_MASSES.copy()
DEUTERATED_MASSES["H"] = 2.01410177812 # + e = 2.01465035812
