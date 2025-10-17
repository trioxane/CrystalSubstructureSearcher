import mendeleev
from typing import Dict, Union

FIRST_100_ELEMENTS = mendeleev.get_all_elements()[:100]

ALLRED_ROCHOW_EN_DICT = {
    el.symbol: el.electronegativity_allred_rochow()
    for el in FIRST_100_ELEMENTS
}

CORDERO_COVALENT_RADIUS_DICT = {
    el.symbol: el.covalent_radius_cordero/100
    if el.covalent_radius_cordero is not None
    else el.covalent_radius_pyykko/100
    for el in FIRST_100_ELEMENTS
}

ALVAREZ_VDW_RADIUS_DICT = {
    el.symbol: el.vdw_radius_alvarez/100
    if el.vdw_radius_alvarez is not None
    else el.vdw_radius/100
    for el in FIRST_100_ELEMENTS
}


def get_element_grouping_dict(grouping: Union[int, Dict] = 1) -> dict:
    """

    mode = 1:
        See "periodic_table" dict

    mode = 2:
        'ENM': 'Al Ga In Sn Tl Pb Bi Nh Fl Mc Lv',
        'EPM': 'Li Be Na Mg K Ca Rb Sr Cs Ba Fr Ra',
        'FM': 'La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb  Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No',
        'MTL': 'B Si Ge As Sb Te Po',
        'NG': 'He Ne Ar Kr Xe Rn Og',
        'NM': 'H C N O F P S Cl Se Br I At Ts',
        'TM': 'Sc Ti V Cr Mn Fe Co Ni Cu Zn  Y Zr Nb Mo Tc Ru Rh Pd Ag Cd  Lu Hf Ta W Re Os Ir Pt Au Hg  Lr Rf Db Sg Bh Hs Mt Ds Rg Cn'

    mode = Dict:
        a dict with element symbols as keys and element groups as values is used for mapping elements
    
    Return: dict
    """

    if isinstance(grouping, dict):
        return {el.symbol: grouping[el.symbol] for el in FIRST_100_ELEMENTS}

    elif grouping == 1:

        arbitrary_types = {
            # Period 1
            "H": "H",  "He": "NG",
            # Period 2
            "Li": "EPM", "Be": "ENM", "B": "MTL", "C": "LNM", "N": "LNM", "O": "LNM", "F": "LNM", "Ne": "NG",
            # Period 3
            "Na": "EPM", "Mg": "EPM", "Al": "ENM", "Si": "MTL", "P": "NM", "S": "NM", "Cl": "NM", "Ar": "NG",
            # Period 4
            "K": "EPM",  "Ca": "EPM",
            "Sc": "TM", "Ti": "TM", "V": "TM", "Cr": "TM", "Mn": "TM", "Fe": "TM", "Co": "TM", "Ni": "TM", "Cu": "TM", "Zn": "TM",
            "Ga": "ENM", "Ge": "MTL", "As": "MTL", "Se": "NM", "Br": "NM",  "Kr": "NG",
            # Period 5
            "Rb": "EPM", "Sr": "EPM",
            "Y": "TM", "Zr": "TM", "Nb": "TM", "Mo": "TM", "Tc": "TM", "Ru": "TM", "Rh": "TM", "Pd": "TM", "Ag": "TM", "Cd": "TM",
            "In": "ENM", "Sn": "ENM", "Sb": "MTL", "Te": "MTL", "I": "NM", "Xe": "NG",
            # Period 6
            "Cs": "EPM", "Ba": "EPM",
            "La": "FM", "Ce": "FM", "Pr": "FM", "Nd": "FM", "Pm": "FM", "Sm": "FM", "Eu": "FM",
            "Gd": "FM", "Tb": "FM", "Dy": "FM", "Ho": "FM", "Er": "FM", "Tm": "FM", "Yb": "FM", "Lu": "FM",
            "Hf": "TM", "Ta": "TM", "W": "TM",  "Re": "TM", "Os": "TM", "Ir": "TM", "Pt": "TM", "Au": "TM", "Hg": "TM",
            "Tl": "ENM", "Pb": "ENM", "Bi": "ENM", "Po": "MTL", "At": "NM", "Rn": "NG",
            # Period 7
            "Fr": "EPM", "Ra": "EPM",
            "Ac": "FM", "Th": "FM", "Pa": "FM", "U": "FM", "Np": "FM", "Pu": "FM", "Am": "FM",
            "Cm": "FM", "Bk": "FM", "Cf": "FM", "Es": "FM", "Fm": "FM", "Md": "FM", "No": "FM", "Lr": "FM"
        }
        return {el.symbol: arbitrary_types[el.symbol] for el in FIRST_100_ELEMENTS}

    elif grouping == 2:

        arbitrary_types = {
            'Actinides': 'FM',
            'Alkali metals': 'EPM',
            'Alkaline earth metals': 'EPM',
            'Halogens': 'NM',
            'Lanthanides': 'FM',
            'Metalloids': 'MTL',
            'Noble gases': 'NG',
            'Nonmetals': 'NM',
            'Poor metals': 'ENM',
            'Transition metals': 'TM'
        }
        return {el.symbol: arbitrary_types[el.series] for el in FIRST_100_ELEMENTS}

    else:
        print('Unknown element grouping')


cH = 1.5
c1, c2, c3, c4, c5 = 2.00, 1.75, 1.75, 1.75, 1.75
c6, c7, c8, c9, c10 = 1.50, 1.25, 1.25, 1.25, 1.25
c11, c12, c13, c14, c15 = 1.25, 1.25, 1.25, 1.25, 1.25
c16, c17, c18, c19, c20 = 1.25, 1.25, 1.20, 1.20, 1.00

MAX_VALENCE_DICT = {
    # Group 1 (Alkali metals) - c1
    "H": cH * 1,   "Li": c1 * 1,  "Na": c1 * 1,  "K": c1 * 1,  "Rb": c1 * 1,  "Cs": c1 * 1,  "Fr": c1 * 1,

    # Group 2 (Alkaline earth metals) - c2
    "Be": c2 * 2,  "Mg": c2 * 2,  "Ca": c2 * 2,  "Sr": c2 * 2,  "Ba": c2 * 2,  "Ra": c2 * 2,

    # Lanthanides - c3
    "La": c3 * 3,  "Ce": c3 * 4,  "Pr": c3 * 5,  "Nd": c3 * 4,  "Pm": c3 * 3,  "Sm": c3 * 3,  "Eu": c3 * 3,
    "Gd": c3 * 3,  "Tb": c3 * 4,  "Dy": c3 * 3,  "Ho": c3 * 3,  "Er": c3 * 3,  "Tm": c3 * 3,  "Yb": c3 * 3,

    # Group 3 - c4
    "Sc": c4 * 3,  "Y": c4 * 3,  "Lu": c3 * 3,

    # Actinides - c5
    "Ac": c4 * 3,  "Th": c5 * 4,  "Pa": c5 * 5,  "U" : c5 * 6,   "Np": c5 * 7,  "Pu": c5 * 6,  "Am": c5 * 6,
    "Cm": c5 * 4,  "Bk": c5 * 4,  "Cf": c5 * 5,  "Es": c5 * 6,   "Fm": c5 * 5,

    # Group 4 - c6
    "Ti": c6 * 4,  "Zr": c6 * 4,  "Hf": c6 * 4,

    # Group 5 - c7
    "V": c7 * 5,   "Nb": c7 * 5,  "Ta": c7 * 5,

    # Group 6 - c8
    "Cr": c8 * 6,  "Mo": c8 * 6,  "W": c8 * 6,

    # Group 7 - c9
    "Mn": c9 * 7,  "Tc": c9 * 7,  "Re": c9 * 7,

    # Group 8 - c10
    "Fe": c10 * 6, "Ru": c10 * 8, "Os": c10 * 8,

    # Group 9 - c11
    "Co": c11 * 5, "Rh": c11 * 6, "Ir": c11 * 6,

    # Group 10 - c12
    "Ni": c12 * 4, "Pd": c12 * 4, "Pt": c12 * 6,

    # Group 11 - c13
    "Cu": c13 * 3, "Ag": c13 * 3, "Au": c13 * 3,

    # Group 12 - c14
    "Zn": c14 * 2, "Cd": c14 * 2, "Hg": c14 * 2,

    # Group 13 - c15
    "B": c15 * 3,  "Al": c15 * 3, "Ga": c15 * 3, "In": c15 * 3, "Tl": c15 * 3,

    # Group 14 - c16
    "C": c16 * 4,  "Si": c16 * 4, "Ge": c16 * 4, "Sn": c16 * 4, "Pb": c16 * 4,

    # Group 15 - c17
    "N": c17 * 5,  "P": c17 * 5,  "As": c17 * 5, "Sb": c17 * 5, "Bi": c17 * 5,

    # Group 16 - c18
    "O": c18 * 2,  "S": c18 * 6,  "Se": c18 * 6, "Te": c18 * 6, "Po": c18 * 6,

    # Group 17 (Halogens) - c19
    "F": c19 * 1,  "Cl": c19 * 7, "Br": c19 * 7, "I": c19 * 7,  "At": c19 * 7,

    # Group 18 (Noble gases) - c20
    "He": c20 * 0.1, "Ne": c20 * 0.1, "Ar": c20 * 0.1, "Kr": c20 * 2, "Xe": c20 * 8, "Rn": c20 * 8
}


def get_max_valence(el_symbol: str) -> float:
    return MAX_VALENCE_DICT[el_symbol]