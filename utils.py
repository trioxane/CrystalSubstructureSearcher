import itertools
import json
from typing import Tuple, Dict, Sequence, Union
from math import gcd
from ast import literal_eval

import pandas as pd
import numpy as np

from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.sites import PeriodicSite


VOLUME_RATIO_THRESHOLD = 10  # threshold for the volume ratio of the transformed cell to the initial unit cell
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"  # letters for WPs
WYCKOFF_CSV_PATH = "./wyckoff_list.csv"  # csv file with WPs data
BV_PARAMETERS_EXCEL_TABLE_PATH = "./BV_estimated_23-04-2024.xlsx"  # excel file with BV parameters


class StructureGraphAnalysisException(Exception):
    """
    Error encountered during crystal graph analysis
    """


class WeirdStructureException(Exception):
    """
    Raised when a weird structure is found, which the LowDimfinder cannot handle
    """


class IntersectingLayeredSubstructuresFound(WeirdStructureException):
    """
    Raised when two crossing layers have been identified as 2-p substructures
    """


class SuperCellSearchError(WeirdStructureException):
    pass


class BulkConnectivityCalculationError(StructureGraphAnalysisException):
    """
    The initial crystal structure graph periodicity is less than 3
    """


class IntraContactsRestorationError(StructureGraphAnalysisException):
    """
    Fragment dimensionality is not preserved during intrafragment contacts restoration
    """


try:
    df_bvparams = pd.read_excel(BV_PARAMETERS_EXCEL_TABLE_PATH, index_col=0).loc[:,
                  ['bond', 'Atom1', 'Atom2', 'confident_prediction',
                   'Rcov_sum', 'delta', 'R0_estimated', 'R0_empirical', 'B']
                  ]
except FileNotFoundError:
    print(f'Excel table with BV parameters has not been found at '
          f'{BV_PARAMETERS_EXCEL_TABLE_PATH}')
else:
    print(f'Excel table with BV parameters has been found at '
          f'{BV_PARAMETERS_EXCEL_TABLE_PATH}')


def calculate_BV(args: tuple[float, str, str]) -> tuple[float, str]:
    """
    Calculate bond valence (BV) for a bond
    between Atom1 (el1) and Atom2 (el2)
    residing at R angstrom from each other

    Args:

        R - interatomic distance;
        el1 - element 1 symbol;
        el2 - element 2 symbol;
    
    Return:
        
        bond_valence, data_source
    """
    R, el1, el2 = args

    empirical_bvs = df_bvparams[
        ((df_bvparams['Atom1'] == el1) & (df_bvparams['Atom2'] == el2)) |
        ((df_bvparams['Atom1'] == el2) & (df_bvparams['Atom2'] == el1))
    ]
    
    if empirical_bvs.shape[0] == 0:
        return np.nan, 'no_estimate'

    if empirical_bvs['R0_empirical'].notna().bool():
        # use R0_empirical
        R0 = empirical_bvs.iat[0, 7]
        B = empirical_bvs.iat[0, 8]
        data_source = 'empirical_and_extrapolated'
    elif empirical_bvs['R0_empirical'].isna().bool():
        # use R0_estimated
        R0 = empirical_bvs.iat[0, 6]
        B = 0.37
        confidence = bool(empirical_bvs.iat[0, 3])
        data_source = f'ML_estimated (confidence: {confidence})'
    else:
        R0 = np.nan
        B = np.nan
        data_source = 'no_estimate'

    return np.exp((R0 - R) / B), data_source


# Helper functions to compute Lattice Transformation Matrix
def ext_gcd(a, b):
    """Extended Euclidean Algorithm to find integers x, y such that ax + by = gcd(a, b)"""
    if b == 0:
        return 1, 0
    else:
        x1, y1 = ext_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return x, y


# Function to calculate the new basis
def calculate_ltm_for_hkl(target_hkl, initial_lattice_vectors) -> Tuple[np.ndarray, float]:
    """
    Calculate a new set of lattice vectors based on the given Miller indices (hkl).

    Parameters:
    - target_hkl: A tuple or list (h, k, l) representing the Miller indices.
    - initial_lattice_vectors: A 3x3 numpy array representing the lattice vectors of the initial unit cell.

    Returns:
    - ltm: A 3x3 numpy array representing the lattice transformation matrix (LTM) for the new basis vectors.

    Raises:
    - ValueError: If the target [hkl] vector is invalid or results in an unrealistic unit cell transformation.
    """

    # Unpack Miller indices
    h, k, l = target_hkl
    if h == 0 and k == 0 and l == 0:
        raise ValueError("The [hkl] Miller indices cannot all be zero.")

    h0, k0, l0 = (h == 0, k == 0, l == 0)  # Check if any index is zero

    # Case 1: Two indices are zero (special orthogonal cases)
    if h0 and k0 or h0 and l0 or k0 and l0:  # If two indices are zero
        if not h0:
            c1, c2, c3 = (0, 1, 0), (0, 0, 1), (1, 0, 0)
        elif not k0:
            c1, c2, c3 = (0, 0, 1), (1, 0, 0), (0, 1, 0)
        elif not l0:
            c1, c2, c3 = (1, 0, 0), (0, 1, 0), (0, 0, 1)

    # Case 2: General case
    else:
        p, q = ext_gcd(k, l)  # Extended GCD of k and l

        # Lattice vectors from the structure
        a1, a2, a3 = initial_lattice_vectors

        # Constants describing the dot product of basis c1 and c2
        k1 = np.dot(p * (k * a1 - h * a2) + q * (l * a1 - h * a3), l * a2 - k * a3)
        k2 = np.dot(l * (k * a1 - h * a2) - k * (l * a1 - h * a3), l * a2 - k * a3)

        # If k2 is not zero, adjust the coefficients
        if abs(k2) > 1e-10:
            i = -int(round(k1 / k2))  # Optimal i for basis vector adjustment
            p, q = p + i * l, q - i * k

        # Extended GCD of (p * k + q * l) and h
        a, b = ext_gcd(p * k + q * l, h)

        # Construct the new basis vectors
        c1 = (p * k + q * l, -p * h, -q * h)
        c2 = np.array((0, l, -k)) // abs(gcd(l, k))
        c3 = (b, a * p, a * q)

    ltm = np.array([c1, c2, c3])

    # Step 5: Calculate the volume ratios for the initial and new unit cells
    # Compute the volume of the initial lattice cell using the determinant of its matrix
    initial_volume = abs(np.linalg.det(initial_lattice_vectors))
    # Compute the volume of the new lattice cell using its vectors
    a_new = (c1[0] * initial_lattice_vectors[0] + c1[1] * initial_lattice_vectors[1] + c1[2] * initial_lattice_vectors[2])
    b_new = (c2[0] * initial_lattice_vectors[0] + c2[1] * initial_lattice_vectors[1] + c2[2] * initial_lattice_vectors[2])
    c_new = (c3[0] * initial_lattice_vectors[0] + c3[1] * initial_lattice_vectors[1] + c3[2] * initial_lattice_vectors[2])
    new_lattice_vectors = np.array([a_new, b_new, c_new])
    new_volume = abs(np.linalg.det(new_lattice_vectors))

    if np.linalg.det(new_lattice_vectors) < 0:
        print("np.linalg.det(new_lattice_vectors) < 0")
        new_lattice_vectors *= -1

    # Ratio of volumes
    volume_ratio = new_volume / initial_volume

    print(f"Initial Volume: {initial_volume:.1f}")
    print(f"New Volume: {new_volume:.1f}")
    print(f"Volume Ratio: {volume_ratio:.3f}")

    if volume_ratio > VOLUME_RATIO_THRESHOLD:
        raise SuperCellSearchError(f"Volume ratio {volume_ratio} exceeds the threshold {VOLUME_RATIO_THRESHOLD}.")

    return ltm, volume_ratio


def find_perpendicular_vector(a_new, lattice_vectors, N) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds a vector that is as close as possible to being perpendicular
    to the given vector (new c-axis), while minimizing its length. The search
    is performed over all linear combinations of lattice vectors with coefficients
    in the range [-N, N].

    Parameters:
    - a_new: The target vector to which the result should be perpendicular (new c-axis).
    - lattice_vectors: A 3x3 numpy array representing the lattice vectors [a, b, c].
    - N: Range of coefficients for linear combinations (-N to N).

    Returns:
    - best_combination: Coefficients (n1, n2, n3) of the best vector combination.
    """
    a, b, c = lattice_vectors
    candidates = []

    # Iterate over all combinations of coefficients in the range [-N, N]
    for n1, n2, n3 in itertools.product(range(-N, N + 1), repeat=3):
        if n1 == 0 and n2 == 0 and n3 == 0:
            continue  # Skip the zero vector

        candidate = n1 * a + n2 * b + n3 * c
        # Cosine of beta, the angle between candidate vector and a_new
        cos_beta = np.dot(candidate, a_new) / (np.linalg.norm(candidate) * np.linalg.norm(a_new))
        cos_beta = np.clip(cos_beta, -1, 1)  # Clip to handle numerical stability
        angle_score = abs(cos_beta)  # Prefer vectors closer to perpendicular with the smallest abs cos_beta
        candidates.append([(n1, n2, n3), candidate, angle_score, np.linalg.norm(candidate)])

    # Convert candidates to numpy array for easier processing
    candidates = np.array(candidates, dtype=object)
    # Scoring: combine angle proximity and vector length
    scores = candidates[:, 2] + candidates[:, 3] / np.max(candidates[:, 3])
    best_index = np.argmin(scores)  # Find index of the best candidate
    best_combination = candidates[best_index, 0]  # Coefficients of the best vector
    best_vector = candidates[best_index, 1]  # Best vector

    return best_combination, best_vector


def calculate_ltm_for_uvw(target_uvw, initial_lattice_vectors, N=2) -> Tuple[np.ndarray, float]:
    """
    Calculate the lattice transformation matrix (LTM) to align an arbitrary [uvw]
    direction with the z-axis ([001]) of the new lattice.

    Parameters:
    - target_uvw: A list or array representing the [u, v, w] direction in the
      crystallographic coordinate system.
    - initial_lattice_vectors: A 3x3 numpy array where each row represents the
      lattice vectors [a, b, c] of the original unit cell.
    - N: Range of coefficients for linear combinations to search for new lattice vectors
      (default is 2).

    Returns:
    - ltm: A 3x3 numpy array representing the lattice transformation matrix (LTM),
      mapping the old lattice vectors to the new coordinate system.
    """

    target_uvw = np.array(target_uvw)

    # Step 1: Define the new c-axis
    c_new = (target_uvw[0] * initial_lattice_vectors[0]
             + target_uvw[1] * initial_lattice_vectors[1]
             + target_uvw[2] * initial_lattice_vectors[2])

    # Step 2: Find the new a-axis (perpendicular to c_new)
    a_combination, a_new = find_perpendicular_vector(c_new, initial_lattice_vectors, N=N)

    # Step 3: Find the new b-axis (perpendicular to both a_new and c_new)
    a_new_norm = a_new / np.linalg.norm(a_new)
    c_new_norm = c_new / np.linalg.norm(c_new)

    b_new_candidates = []
    for n1, n2, n3 in itertools.product(range(-N, N + 1), repeat=3):
        if n1 == 0 and n2 == 0 and n3 == 0:
            continue

        candidate = (n1 * initial_lattice_vectors[0]
                     + n2 * initial_lattice_vectors[1]
                     + n3 * initial_lattice_vectors[2])
        # Compute angles b_new to c_new (alpha) and b_new to a_new (gamma)
        alpha = np.rad2deg(np.arccos(np.clip(np.dot(candidate, c_new_norm) / np.linalg.norm(candidate), -1, 1)))
        gamma = np.rad2deg(np.arccos(np.clip(np.dot(candidate, a_new_norm) / np.linalg.norm(candidate), -1, 1)))

        # Ensure the candidate is approximately perpendicular to both
        # if 60 < abs(alpha) < 120 and 60 < abs(gamma) < 120: TODO think about selection criteria
        if 75 < abs(alpha) < 120 and 75 < abs(gamma) < 120:
            b_new_candidates.append(((n1, n2, n3), candidate, np.linalg.norm(candidate)))

    # Choose the shortest valid b_new candidate
    b_combination, b_new, _ = min(b_new_candidates, key=lambda x: x[2])

    # Step 4: Construct the lattice transformation matrix
    ltm = np.array([a_combination, b_combination, target_uvw])

    # Step 5: Calculate the volume ratios for the initial and new unit cells
    # Compute the volume of the initial lattice cell using the determinant of its matrix
    initial_volume = abs(np.linalg.det(initial_lattice_vectors))
    # Compute the volume of the new lattice cell using its vectors
    new_lattice_vectors = np.array([a_new, b_new, c_new])
    new_volume = abs(np.linalg.det(new_lattice_vectors))

    # Ratio of volumes
    volume_ratio = new_volume / initial_volume

    print(f"Initial Volume: {initial_volume:.1f}")
    print(f"New Volume: {new_volume:.1f}")
    print(f"Volume Ratio: {volume_ratio:.3f}")

    if volume_ratio > VOLUME_RATIO_THRESHOLD:
        raise SuperCellSearchError(f"Volume ratio {volume_ratio} exceeds the threshold {VOLUME_RATIO_THRESHOLD}.")

    return ltm, volume_ratio


    # # Let's make sure we have a left-handed crystallographic system
    # if np.linalg.det(slab_scale_factor) < 0:
    #     slab_scale_factor *= -1


def calculate_area(oriented_cell: np.ndarray) -> float:

    a = oriented_cell[0]
    b = oriented_cell[1]

    S = np.linalg.norm(np.cross(a, b))

    return S


def get_zone_axis(h1k1l1: Sequence, h2k2l2: Sequence) -> Union[None, np.ndarray]:
    """
    Determine the zone axis [uvw] from the intersection of two planes using the crystallographic zone law.

    Args:
        h1k1l1: A tuple or list (h1, k1, l1) representing the Miller indices of the first plane.
        h2k2l2: A tuple or list (h2, k2, l2) representing the Miller indices of the second plane.

    Returns:
        A numpy array representing the zone axis [u, v, w], or None if the planes are parallel.
    """

    # Compute the cross product to determine the intersection direction
    zone_axis = np.cross(h1k1l1, h2k2l2)

    # If the result is a zero vector, the planes are parallel
    if np.allclose(zone_axis, 0):
        return None

    # Convert to smallest integer values by dividing by GCD
    gcd = np.gcd.reduce(zone_axis)
    if gcd != 0:
        zone_axis = (zone_axis / gcd).astype(int)

    return zone_axis


def get_wyckoffs_dict(space_group_number: int) -> Dict:
    """
    Loads Wyckoff positions for a given space group number from a CSV file.

    Args:
        space_group_number (int): The international space group number.

    Returns:
        dict: A dict of Wyckoff positions, containing a list of SymmOp operations.
    """

    # Load Wyckoff position data from CSV
    df = pd.read_csv(WYCKOFF_CSV_PATH, index_col=0)

    # Extract the Wyckoff positions for the given space group number
    wyckoff_strings = literal_eval(df.loc[space_group_number, "0"])  # Convert string to list

    wyckoffs = []
    for wp_group in wyckoff_strings:
        wyckoff_ops = [SymmOp.from_xyz_string(op) for op in wp_group]
        wyckoffs.append(wyckoff_ops)

    length = len(wyckoffs)
    wyckoffs_dict = {}
    for i in range(len(wyckoffs)):
        mult = len(wyckoffs[i])
        letter = LETTERS[length - 1 - i]
        wyckoffs_dict[f"{mult}{letter}"] = wyckoffs[i]

    # reverse dict
    wyckoffs_dict = {k: v for k, v in list(wyckoffs_dict.items())[::-1]}

    return wyckoffs_dict


def get_wyckoff_position_from_xyz(wyckoffs_dict: Dict, xyz: Sequence, decimals: int = 4) -> str:
    """
    Determines the Wyckoff position of a given fractional coordinate.

    Args:
        wyckoffs_dict (int): Dictionary with WPs for a given Space Group.
        xyz (tuple or np.ndarray): Fractional coordinate [x, y, z].
        decimals (int): Number of decimals for rounding.

    Returns:
        list: The Wyckoff position (list of SymmOp) if found, otherwise None.
    """
    xyz = np.round(np.array(xyz, dtype=float), decimals=decimals)
    xyz -= np.floor(xyz)
    # Iterate over Wyckoff positions and check if the point belongs
    for letter, wp_ops in wyckoffs_dict.items():
        # Apply all symmetry operations in the Wyckoff position to xyz
        orbit = np.array([op.operate(xyz) for op in wp_ops])
        orbit -= np.floor(orbit)

        # Check if the transformed points match the original point
        if np.any(np.all(np.isclose(orbit, xyz, atol=1e-4), axis=1)):
            # Ensure all transformed points are unique
            if len(orbit) == len(np.unique(orbit.round(decimals=decimals), axis=0)):
                return letter  # Return the matching Wyckoff position letter
    return '--'
    # raise RuntimeError(f"Cannot find the suitable Wyckoff position for the given input position {xyz}")


def determine_wyckoff_position(
    frac_coords: Tuple[float, float, float], struct: Structure, symprec=0.01, dist_tol=0.01
) -> str:
    """
    Determines the Wyckoff position of a given fractional coordinate in a unit cell.
    Based on https://github.com/SMTG-Bham/doped/blob/a4ea4c8a8a206785cba1f5d9f9db86aee6919b76/doped/utils/symmetry.py#L328C5-L328C16

    Args:
        frac_coords (Tuple[float, float, float]): Fractional coordinates of the point.
        struct (Structure): pymatgen Structure object corresponding to the unit cell.
        symprec (float, optional): Symmetry precision for SpacegroupAnalyzer (default: 0.01).
        dist_tol (float, optional): Distance tolerance for equivalent site determination (default: 0.01).

    Returns:
        str: The Wyckoff position label (e.g., "4a", "2b", etc.).
    """

    try:
        # Step 1: Compute symmetry operations
        sga = SpacegroupAnalyzer(struct, symprec=symprec)
        symm_ops = sga.get_symmetry_operations()

        # Step 2: Get all equivalent sites for the input fractional coordinates
        unique_sites = []
        dummy_site = PeriodicSite("X", frac_coords, struct.lattice)  # Dummy atom to track symmetry effects

        for symm_op in symm_ops:
            transformed_site = symm_op.operate(frac_coords)
            transformed_site = np.mod(transformed_site, 1)  # Bring into unit cell

            # Check if this site is unique
            if not unique_sites or np.linalg.norm(
                np.dot(
                    np.array([site.frac_coords for site in unique_sites]) - transformed_site,
                    struct.lattice.matrix,
                ), axis=-1
            ).min() > dist_tol:
                unique_sites.append(PeriodicSite("X", transformed_site, struct.lattice))

        # Step 3: Create a new structure with the equivalent sites added
        struct_with_sites = Structure(
            struct.lattice,  # Keep the same lattice
            [site.specie for site in struct.sites] + [site.specie for site in unique_sites],  # Combine species
            [site.frac_coords for site in struct.sites] + [site.frac_coords for site in unique_sites],  # Combine fractional coords
            to_unit_cell=True  # Ensure sites remain inside the unit cell
        )

        # Step 4: Recompute symmetry dataset with additional sites
        sga_with_all_sites = SpacegroupAnalyzer(struct_with_sites, symprec=symprec)
        symm_dataset = sga_with_all_sites.get_symmetry_dataset()

        # Step 5: Compute Wyckoff multiplicity & label
        conv_cell_factor = len(symm_dataset["std_positions"]) / len(symm_dataset["wyckoffs"])
        multiplicity = int(conv_cell_factor * len(unique_sites))
        wyckoff_label = f"{multiplicity}{symm_dataset['wyckoffs'][-1]}"
    except Exception as e:
        print(e.__class__.__name__)
        wyckoff_label = '--'

    return wyckoff_label


def get_weighted_mean(seq: Sequence[Tuple[float, float]]) -> float:
    """
    Computes the weighted mean of a sequence of (value, weight) pairs.

    Args:
        seq (Sequence[Tuple[float, float]]): A sequence of tuples where each tuple contains (value, weight).

    Returns:
        float: The weighted mean of the input sequence.

    Raises:
        ValueError: If the total weight is zero to avoid division by zero.
    """
    numerator = sum(weight for _, weight in seq)

    if numerator == 0:
        raise ValueError("Total weight must not be zero when computing weighted mean.")

    weighted_mean = sum(value * (weight / numerator) for value, weight in seq)
    return weighted_mean


class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
