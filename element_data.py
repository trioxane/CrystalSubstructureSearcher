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
