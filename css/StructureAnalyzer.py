import itertools
from typing import List, Dict, Union, Optional
import copy
from pathlib import Path
from collections import defaultdict

import numpy as np

from pymatgen.core import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.dimensionality import get_dimensionality_larsen, get_structure_components

from . import utils
from .element_data import ALLRED_ROCHOW_EN_DICT, CORDERO_COVALENT_RADIUS_DICT, ALVAREZ_VDW_RADIUS_DICT
from .structure_classes import Substructure, CrystalSubstructures, TargetSubstructure, Contact

from time import time


class CrystalSubstructureSearcher:
    """
    A class for analyzing crystal structures using graph-based methods.

    Args:
        file_name (str): The name of the CIF file.
        connectivity_calculator (VoronoiNN): An object of the VoronoiNN class for computing crystal connectivity.
        bond_property (str): The selected bond property (one of the 'R', 'SA', 'A', 'BV', 'PI') to be used
                             as the weight of edges in the crystal graph.

    Attributes:
        structure (Structure): The crystal structure.
        _connectivity_calculator (VoronoiNN): The VoronoiNN object for connectivity calculations.
        bond_property (str): The selected bond property.

    Methods:
        analyze_graph(target_periodicity: int) -> CrystalSubstructures:
            Editing the crystal graph by iteratively removing edges based on the provided border weights.
    """

    def __init__(self, file_name: str, connectivity_calculator: VoronoiNN, bond_property: str = 'BV',
                 target_periodicity: int = 2) -> None:
        """
        Initializes the CrystalSubstructureSearcher class.

        Args:
            file_name (str): The name of the input CIF file.
            connectivity_calculator (VoronoiNN): An instance of VoronoiNN for computing crystal connectivity.
            bond_property (str): The selected bond property ('R', 'SA', 'A', 'BV', 'PI') used for weighting edges in the graph.
            target_periodicity (int): The target periodicity of the substructure.

        Attributes:
            target_periodicity (int): The target periodicity of the substructure.
            crystal_graph_name (str): The name of the crystal structure graph.
            bond_property (str): The bond property selected for edge weighting.
            structure (Structure): The crystal structure object.
            _connectivity_calculator (VoronoiNN): VoronoiNN instance for connectivity calculations.
            sg (StructureGraph): Graph representation of the structure.
            deleted_contacts (Optional[Dict]): Stores deleted interatomic contacts.
            monitor (Optional[List]): Stores analysis progress data.
            suspicious_contacts (set): Tracks contacts with unusually high bond valence (BV).
            restored_crystal_substructures (Optional[CrystalSubstructures]): Stores restored substructures.
            _substructure_periodicities (set): Tracks identified substructure periodicities in the graph.
            _ltm_is_identified (bool): Indicates whether a lattice transformation matrix is identified.
            _ltm (tuple): Stores the lattice transformation matrix.
            _atom_valences (Optional[defaultdict]): Store atom valences.
            _intercomponent_contacts_in_original_cell (Optional[List[Contact]]): List of Contacts objects corresponding
                to inter contacts in the original cell

        The method also initializes the crystal graph without lattice transformation.
        """

        self.target_periodicity = target_periodicity
        self.crystal_graph_name = Path(file_name).stem
        self.bond_property = bond_property
        self._initial_cell: Optional[Structure] = None  # original structure
        self.structure = self._read_in_structure(file_name)
        self._connectivity_calculator = connectivity_calculator

        self.sg = None  # StructureGraph instance
        self.deleted_contacts = None  # Stores deleted interatomic contacts
        self.monitor = None  # Tracks analysis progress
        self.suspicious_contacts = set()  # Stores suspiciously high BV contacts
        self.restored_crystal_substructures = None  # Stores restored substructures after graph editing
        self._substructure_periodicities = set()  # Tracks encountered substructure periodicities
        self._ltm_is_identified = False  # Flag for lattice transformation matrix identification
        self._ltm = (1, 1, 1)  # Stores lattice transformation matrix
        self._intercomponent_contacts_in_original_cell: Optional[List[Contact]] = None
        self._atom_valences: Optional[defaultdict] = None  # store atom valences

        # Creates an initial structure graph without lattice transformation
        self._create_structure_graph(transform_lattice=False, calculate_valences=False)

    def _read_in_structure(self, file_name: str) -> Structure:
        """
        Reads the crystal structure from a CIF file and converts it into its conventional cell.

        Args:
            file_name (str): Path to the CIF file.

        Returns:
            Structure: The conventional standard crystal structure.

        The method performs the following steps:
        - Reads the primitive cell from the CIF file.
        """
        read_in_structure = Structure.from_file(file_name, primitive=False)
        # read_in_structure.to('readin.cif')

        print(
            f"\n{'#'*60}\nInput structure filename: {self.crystal_graph_name}\n"
            f"composition: {read_in_structure.composition.formula.replace(' ', '')}\n"
            # f"space group: {read_in_structure.get_space_group_info()[0]}\n"
            # f"cell a_b_c: {read_in_structure.lattice.abc}\n"
            # f"cell alpha_beta_gamma: {read_in_structure.lattice.angles}\n"
        )

        self._initial_cell = read_in_structure.copy()
        return read_in_structure

    def _add_weight_to_graph_edges(self, site_idx: int, neighbor_data: Dict, calculate_valences: bool,
                                   suspicious_BV_threshold: float = 5.0) -> None:
        """
        Adds an edge weight (A, R, SA, PI, BV) to the crystal graph based on VoronoiNN calculations.
        Article on PI: https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc02238b

        Args:
            site_idx (int): Index of the central site in the structure.
            neighbor_data (Dict): Information about the neighboring site, including polyhedral properties.
            suspicious_BV_threshold (float, optional): Threshold above which bond is considered suspicious
                (set as max quintuple bond Cr-Cr/Mo-Mo).

        The method performs the following steps:
        - Extracts atomic symbols of the central and neighboring atoms.
        - Computes different edge properties such as area (A), bond distance (R), solid angle (SA), and Penetration Index (PI).
        - Makes bond valence (BV) calculations and checks if the value is suspiciously high.
        - Adds an edge to the structure graph with the selected bond property as the weight.
        """

        # Extract atomic symbols
        central_atom = self.structure[site_idx].specie.symbol
        neighbour_atom = neighbor_data['site'].specie.symbol

        # Compute edge properties based on Voronoi analysis
        A = neighbor_data['poly_info']['area']  # Face area
        R = neighbor_data['poly_info']['face_dist'] * 2  # Interatomic distance
        SA = neighbor_data['poly_info']['solid_angle']  # Solid angle contribution

        # Compute Penetration Index based on atomic radii and bond length
        PI = 100 * (ALVAREZ_VDW_RADIUS_DICT[central_atom] + ALVAREZ_VDW_RADIUS_DICT[neighbour_atom] - R) \
             / ((ALVAREZ_VDW_RADIUS_DICT[central_atom] - CORDERO_COVALENT_RADIUS_DICT[central_atom]) +
                (ALVAREZ_VDW_RADIUS_DICT[neighbour_atom] - CORDERO_COVALENT_RADIUS_DICT[neighbour_atom]))

        # Compute bond valence (BV) using utility function
        BV, BV_calc_method = utils.calculate_BV((R, central_atom, neighbour_atom))
        BV = round(BV, 5) + 1e-6  # Round for consistency and add minuscule weight for stability
        # ??? weight 0.0 for StructureGraph not acceptable ???

        # If BV is suspiciously high, add it to the suspicious contacts list
        if BV > suspicious_BV_threshold:
            self.suspicious_contacts.add(f'{central_atom}-{neighbour_atom} {R:.3f}A {BV:.1f} vu')

        # Store computed edge properties
        edge_properties = {'BV': BV, 'BV_calc_method': BV_calc_method, 'R': R, 'A': A, 'SA': SA, 'PI': PI}

        # Add an edge to the structure graph with the selected bond property as the weight
        self.sg.add_edge(
            from_index=site_idx,
            from_jimage=(0, 0, 0),
            to_index=neighbor_data["site_index"],
            to_jimage=neighbor_data["image"],
            edge_properties=edge_properties,
            weight=edge_properties[self.bond_property],  # Use selected bond property as the edge weight
            warn_duplicates=False,
        )

        if calculate_valences:
            self._atom_valences[site_idx] += BV

    def _create_structure_graph(self, transform_lattice: bool = False, calculate_valences: bool = False, N: int = 2) -> None:
        """
        Creates a StructureGraph instance from the Structure by calculating connectivity
        and adding attributes to the crystal structure graph edges.

        Args:
            transform_lattice (bool, optional): If True, transforms the lattice using LTM before constructing the graph.
            calculate_valences (bool, optional): If True store data on atom valences.
            N (int, optional): Determines the extension of the cell into supercell. N=2 is usually enough,
            but in some cases larger N may be required if the substructure extends over multiple unit cells, e.g.
            in ICSD 196 - 1-p substructure, ICSD 195327 - 2-p substructure.

        The method performs the following steps:
        - If lattice transformation is requested, it applies the transformation if identified.
        - Creates an empty StructureGraph instance.
        - Iterates over all neighboring sites to add weighted edges to the graph.
        - Adds atomic properties (element symbol, electronegativity) to graph nodes.
        - Ensures that the periodicity of the graph is at least 3 (bulk material).

        Raises:
            utils.BulkConnectivityCalculationError: If the graph periodicity is found to be less than 3.
        """

        if calculate_valences:
            self._atom_valences = defaultdict(float)  # dict with valences of atoms in the structure

        # Apply lattice transformation if requested and identified
        if transform_lattice:

            if not self._ltm_is_identified:
                print(f'Lattice will NOT be transformed')
            else:
                print(f'Lattice will be transformed')
                # print(f'Total sites before: {self.structure.num_sites}')
                # print('LTM:')
                # print(f'{self._ltm}')

                # Apply transformation to the structure
                self.structure.make_supercell(self._ltm)
                # print(f'Total sites after 1st transformation: {self.structure.num_sites}')

                # Additional expansion depending on target periodicity
                if self.target_periodicity == 2:
                    self.structure.make_supercell((1, 1, N))  # Expand along c direction
                elif self.target_periodicity == 1:
                    self.structure.make_supercell((N, N, 1))  # Expand along a and b directions
                elif self.target_periodicity == 0:
                    self.structure.make_supercell((N, N, N))  # Expand in all three directions

                # print(f'Lattice was transformed')
                # print(f'Total sites after 2nd transformation: {self.structure.num_sites}')

        # Initialize an empty StructureGraph instance
        self.sg = StructureGraph.with_empty_graph(self.structure, name=f"CRYSTAL_GRAPH_{self.crystal_graph_name}")

        # Iterate over each site and compute its neighbors using VoronoiNN
        for site_idx, site_neighbors in enumerate(self._connectivity_calculator.get_all_nn_info(self.structure)):
            for site_neighbor in site_neighbors:
                self._add_weight_to_graph_edges(site_idx, site_neighbor, calculate_valences=calculate_valences)

        # Add element properties to graph nodes
        for i, node in enumerate(self.sg.graph.nodes):
            node_element_symbol = self.sg.structure[i].specie.symbol
            self.sg.graph.nodes[i]['element'] = node_element_symbol
            self.sg.graph.nodes[i]['EN'] = round(ALLRED_ROCHOW_EN_DICT[node_element_symbol], 6)
            if calculate_valences:
                self.sg.graph.nodes[i]['valence'] = round(self._atom_valences[i], 2)

        # Check if the periodicity of the structure graph is valid
        structure_graph_periodicity = get_dimensionality_larsen(self.sg)
        if structure_graph_periodicity < 3:
            raise utils.BulkConnectivityCalculationError(
                (f'(!!!) The crystal structure graph periodicity is {structure_graph_periodicity} < 3 (!!!)\n'
                 f'Possible reasons: the crystal structure might be unreasonable, or\n'
                 f'consider decreasing the `tol` parameter in the VoronoiNN instance\n'
                 f'to allow more interatomic contacts to be identified')
            )

    def _calculate_SG_BV_sum(self, sg: StructureGraph) -> float:
        """
        Calculates the total bond valence sum (BVS) for a given structure graph.

        Args:
            sg (StructureGraph): The structure graph for which bond valence sum is computed.

        Returns:
            float: The total sum of bond valences in the graph.

        The method iterates over all edges in the structure graph and sums up
        the bond valence (BV) values associated with each edge.
        """
        return sum([edge_data[2] for edge_data in sg.graph.edges(data='BV')])

    def _get_threshold_weights(self) -> np.ndarray:
        """
        Returns a sorted list of threshold values for graph editing by cutting edges
        with weights lower than the threshold.

        Args:
            merge_close_weights (bool, optional): If True, merges weights that are very close together.

        Returns:
            np.ndarray: A sorted array of unique threshold weights.

        The method performs the following steps:
        - Defines the number of decimal places to round the weights based on bond property.
        - Sorts and rounds unique weights from the structure graph.

        Notes:
            - `BV`, `SA`, `A`, and `PI` have stronger bonding for higher values.
            - `R` (bond length) has stronger bonding for lower values.
        """
        DECIMALS = {'R': 3, 'BV': 4, 'SA': 3, 'A': 2, 'PI': 2}  # TODO check accuracy for R, SA, A, PI ...?

        decimals = DECIMALS[self.bond_property]
        delta = 10**-decimals

        # Determine sorting order: reverse for 'R' (bond length), normal for others
        if self.bond_property in ('BV', 'SA', 'A', 'PI'):
            reverse = False  # higher values correspond to stronger bonding
            delta = delta
        else:  # for 'R'
            reverse = True  # lower values (shorter bond lengths) correspond to stronger bonding
            delta = -delta

        # Extract and sort unique bond property weights
        unique_weights = set(np.round(self.sg.weight_statistics['all_weights'], decimals=decimals))
        unique_weights = np.array(sorted(unique_weights, reverse=reverse))
        # add small delta for them to serve as thresholds
        unique_weights += delta

        return unique_weights

    def _check_plane_crossing(self, target_substructure_sg, verbose=False) -> None:

        _2p_substructures_orientations = set()
        for i, component in enumerate(
                get_structure_components(target_substructure_sg, inc_orientation=True, inc_site_ids=True)
        ):
            periodicity = component['dimensionality']
            orientation = component['orientation'] if component['orientation'] is not None else '_'
            composition = component['structure_graph'].structure.composition.formula.replace(" ", "")
            if verbose:
                print(f'TARGET SUBSTRUCTURE {periodicity}-p component {i}: {orientation}, {composition}')

            if periodicity == 2:
                _2p_substructures_orientations.add(orientation)

        if len(_2p_substructures_orientations) == 1:
            pass
        else:
            # if otherwise there several layers we must check if they cross
            for h1k1l1, h2k2l2 in itertools.combinations(_2p_substructures_orientations, 2):
                zone_axis = utils.get_zone_axis(h1k1l1, h2k2l2)
                if zone_axis is not None:
                    raise utils.IntersectingLayeredSubstructuresFound(
                        f"layers in planes {h1k1l1} and {h2k2l2} intersect along {zone_axis}"
                    )

        return None

    def _restore_intra_contacts1(self, low_periodic_substructures: List[Dict]) -> defaultdict[int, List[Substructure]]:
        """
        Restores intra contacts that were previously removed while reducing periodicity.
        The restoration ensures that the periodicity of the substructure does not increase.

        Args:
            low_periodic_substructures (List[Dict]): A list of dictionaries containing structure graphs with reduced periodicity.

        Returns:
            defaultdict[int, List[Substructure]]: A dictionary mapping periodicity to lists of restored substructures.

        The method performs the following steps:
        - Stores the original 3-periodic structure graph.
        - Iterates over each low-periodic substructure and attempts to restore broken edges.
        - Adds back intra contacts while ensuring that periodicity does not increase.

        Raises:
            utils.IntraContactsRestorationError: If the periodicity changes unexpectedly after restoration.
        """

        # store original 3-periodic structure graph
        substructures = defaultdict(list)
        substructures[3] = [
            Substructure(
                max_periodicity=3,
                sg=self.sg,
                deleted_contacts=None,
                inter_contacts=set(),
                BVS=self._calculate_SG_BV_sum(self.sg)
            )
        ]

        # print('\nRestoring intracomponent contacts')
        # print(f'NUM_CONTACTS in original graph: {len(self.sg.graph.edges())}')
        for n, low_p_substructure in enumerate(low_periodic_substructures):

            for i, component in enumerate(
                    get_structure_components(low_p_substructure['sg'], inc_orientation=True, inc_site_ids=True)
            ):
                periodicity = component['dimensionality']
                orientation = component['orientation'] if component['orientation'] is not None else '_'
                composition = component['structure_graph'].structure.composition.formula.replace(" ", "")
                fragment_sites = component['site_ids']
                # print(f'Substructure {n+1}, {periodicity}-p component {i}: {orientation}, {composition}')

            # BV sum BEFORE restoring INTRA fragment contacts
            _edited_graph_total_bvs = self._calculate_SG_BV_sum(low_p_substructure['sg'])

            tic = time()
            # print(f"NUM_CONTACTS in the edited graph: {len(low_p_substructure['sg'].graph.edges())}")
            # print(f"TOTAL_BVS in the edited graph: {_edited_graph_total_bvs:.3f}")

            test_graph_1 = copy.copy(low_p_substructure['sg'])
            # Try to restore contacts iteratively starting from the strongest ones
            # TODO check this part for edge weights other than BV !!! like R etc.
            for _, broken_edges in sorted(
                    low_p_substructure['deleted_contacts'].items(), key=lambda kv: kv[0][1], reverse=True
            ):
                test_graph_2 = copy.copy(test_graph_1)
                for edge in broken_edges:
                    (n1, n2, translation), edge_data = edge
                    test_graph_2.add_edge(
                        from_index=n1,
                        to_index=n2,
                        from_jimage=(0, 0, 0),
                        to_jimage=translation,
                        edge_properties={k: v for k, v in edge_data.items() if k != 'to_jimage'},
                    )

                # Ensure periodicity does not increase after restoring contacts
                if get_dimensionality_larsen(test_graph_2) <= low_p_substructure['max_periodicity']:
                    test_graph_1 = test_graph_2  # Accept restored contacts
                else:
                    test_graph_1 = test_graph_1  # Reject if periodicity increases

            low_p_substructure['sg'] = test_graph_1
            inter_contacts = self.sg.diff(test_graph_1)['self']

            # here we get the info on the inter contacts in the !!! original cell !!! of the structure
            # to calculate some descriptors using original lattice
            if get_dimensionality_larsen(test_graph_1) == self.target_periodicity:

                intercomponent_contacts = []
                for contact_type, contacts_list in low_p_substructure['deleted_contacts'].items():
                    for contact in contacts_list:
                        if contact[0] in inter_contacts:

                            intercomponent_contacts.append(
                                Contact.from_data(
                                    pmg_structure=self.sg.structure,
                                    at1_idx=contact[0][0],
                                    at2_idx=contact[0][1],
                                    t=contact[0][2],
                                    contact_characteristics=contact[1],
                                    element_grouping_dict=None,
                                    identify_WP=True,
                                )
                            )
                self._intercomponent_contacts_in_original_cell = intercomponent_contacts

            # Ensure periodicity is preserved after restoration
            if get_dimensionality_larsen(low_p_substructure['sg']) != low_p_substructure['max_periodicity']:
                raise utils.IntraContactsRestorationError(
                    'Substructure periodicity is not preserved during intra contacts restoration'
                )

            # BV sum AFTER restoring INTRA fragment contacts
            _restored_graph_total_bvs = self._calculate_SG_BV_sum(low_p_substructure['sg'])

            toc = time()
            # print(f"NUM_CONTACTS in the restored graph: {len(low_p_substructure['sg'].graph.edges())}")
            # print(f"TOTAL_BVS in the restored graph: {_restored_graph_total_bvs:.3f}")
            # print(f"NUM_INTER_CONTACTS: {len(inter_contacts)}")
            # print(f"TIME required to restore graph: {toc - tic:.1f} sec\n")

            # Store the restored substructure
            substructures[low_p_substructure['max_periodicity']].append(
                Substructure(
                    max_periodicity=low_p_substructure['max_periodicity'],
                    sg=low_p_substructure['sg'],
                    deleted_contacts=low_p_substructure['deleted_contacts'],
                    inter_contacts=inter_contacts,
                    BVS=self._calculate_SG_BV_sum(low_p_substructure['sg'])
                )
            )

        return substructures

    def _restore_intra_contacts2(self, target_substructure: Dict) -> Optional[TargetSubstructure]:
        """
        Restores intra contacts for the target substructure while ensuring periodicity does not increase.

        Args:
            target_substructure (Dict): A dictionary containing the structure graph of the target substructure.

        Returns:
            TargetSubstructure: The restored target substructure with updated connectivity.

        The method performs the following steps:
        - Iterates over structure components to analyze their periodicity and composition.
        - Restores broken intra-fragment contacts iteratively while ensuring that periodicity does not increase.
        - Computes bond valence sum (BVS) before and after restoration.
        - Outputs details of the restored graph, including the number of contacts and time taken.

        Raises:
            utils.IntraContactsRestorationError: If the periodicity changes unexpectedly after restoration.
        """
        # print('\nRestoring intracomponent contacts in TARGET SUBSTRUCTURE')

        # BV sum BEFORE restoring INTRA fragment contacts
        _edited_graph_total_bvs = self._calculate_SG_BV_sum(target_substructure['sg'])

        tic = time()
        # print(f"NUM_CONTACTS in the edited graph: {len(target_substructure['sg'].graph.edges())}")
        # print(f"TOTAL_BVS in the edited graph: {_edited_graph_total_bvs:.3f}")

        # first check if there are no crossing 2-p substructures (like in CoU6 ICSD 108323)
        self._check_plane_crossing(target_substructure['sg'])

        test_graph_1 = copy.copy(target_substructure['sg'])

        # Try to restore contacts iteratively
        for threshold, broken_edges in sorted(
                target_substructure['deleted_contacts'].items(), key=lambda kv: kv[0][1], reverse=True
        ):
            test_graph_2 = copy.copy(test_graph_1)
            for edge in broken_edges:
                (n1, n2, translation), edge_data, _ = edge
                test_graph_2.add_edge(
                    from_index=n1,
                    to_index=n2,
                    from_jimage=(0, 0, 0),
                    to_jimage=translation,
                    edge_properties={k: v for k, v in edge_data.items() if k != 'to_jimage'},
                )

            # Ensure periodicity does not increase after restoring contacts
            # TODO ICSD 29 after restoration of bonds the two 1-p components merge into single 2-p ones
            #  consider another restoration criteria
            if get_dimensionality_larsen(test_graph_2) <= self.target_periodicity:
                test_graph_1 = test_graph_2  # Accept restored contacts
            else:
                test_graph_1 = test_graph_1  # Reject if periodicity increases

        target_substructure['sg'] = test_graph_1

        # check again if there are no crossing 2-p substructures (like in CoU6 ICSD 108323)
        self._check_plane_crossing(target_substructure['sg'], verbose=True)

        # Ensure periodicity is preserved after restoration
        if get_dimensionality_larsen(target_substructure['sg']) != self.target_periodicity:
            raise utils.IntraContactsRestorationError(
                'Substructure periodicity is not preserved during intra contacts restoration'
            )

        # BV sum AFTER restoring INTRA fragment contacts
        _restored_graph_total_bvs = self._calculate_SG_BV_sum(target_substructure['sg'])

        toc = time()
        # print(f"NUM_CONTACTS in the restored graph: {len(target_substructure['sg'].graph.edges())}")
        # print(f"TOTAL_BVS in the restored graph: {_restored_graph_total_bvs:.3f}")
        # print(f"NUM_INTER_CONTACTS: {len(self.sg.diff(target_substructure['sg'])['self'])}")
        # print(f"TIME required to restore graph: {toc - tic:.1f} sec")

        # Return the restored target substructure
        restored_target_crystal_substructure = TargetSubstructure(
            max_periodicity=self.target_periodicity,
            sg=target_substructure['sg'],
            deleted_contacts=target_substructure['deleted_contacts'],
            inter_contacts=self.sg.diff(target_substructure['sg'])['self'],
            intercomponent_contacts_in_original_cell=self._intercomponent_contacts_in_original_cell,
            BVS=self._calculate_SG_BV_sum(target_substructure['sg']),
            total_BVS=target_substructure['total_BVS'],
            ltm_used=self._ltm,
            initial_cell=self._initial_cell,
        )

        return restored_target_crystal_substructure

    def _determine_lattice_transformation(self, low_periodic_substructure_graphs: List[Dict]) -> None:
        """
        Determines the lattice transformation matrix (LTM) required to get the unit cell with
        layers lying in (001) plane and rods directed in [001] direction

        Args:
            low_periodic_substructure_graphs (List[Dict]): A list of dictionaries containing
                                                           structure graphs with reduced periodicity.

        The method performs the following steps:
        - Checks if the target periodicity substructure is found; if not, sets default transformation.
        - Iterates over the components of low-periodic substructures to determine the LTM.
        - Computes transformation matrices based on target periodicity.
        - Selects the LTM that maximizes volume preservation.

        Raises:
            None. If the target periodicity is not found, it defaults to an identity transformation.
        """

        # Ensure we have got substructure with target periodicity before proceeding
        if not self._substructure_periodicities or self.target_periodicity not in self._substructure_periodicities:
            print(f'(!!!) Substructure of target periodicity {self.target_periodicity} was not found (!!!)')
            self._ltm_is_identified = False
            return None

        results = []
        # Iterate over low-periodic substructures to determine the best transformation matrix
        for i, substructure_dict in enumerate(low_periodic_substructure_graphs, start=1):
            for component in get_structure_components(substructure_dict['sg'], inc_orientation=True, inc_site_ids=False):
                periodicity = component['dimensionality']
                orientation = component['orientation']
                hkl_uvw = np.array(orientation) if orientation is not None else 0
                # print(f'Substructure {i}, component {periodicity}-p {orientation}')

                # Compute transformation matrix based on target periodicity
                if periodicity == self.target_periodicity:

                    # check target periodicity component count here
                    if self.target_periodicity == 2:
                        result = utils.calculate_ltm_for_hkl(
                            target_hkl=hkl_uvw, initial_lattice_vectors=self.sg.structure.lattice.matrix
                        )
                    elif self.target_periodicity == 1:
                        result = utils.calculate_ltm_for_uvw(
                            target_uvw=hkl_uvw, initial_lattice_vectors=self.sg.structure.lattice.matrix
                        )
                    else:
                        result = (np.array([1, 1, 1]), 8.0)  # Default transformation

                    if result is None:
                        continue  # Skip if no transformation is found

                    ltm, V_new_to_V_old = result
                    results.append((ltm, V_new_to_V_old))

        # Select the LTM that maximizes volume
        self._ltm_is_identified = True
        self._ltm = max(results, key=lambda v: v[1])[0] if results else (2, 2, 2)

        return None

    def _analyze_graph1(self) -> Optional[defaultdict]:
        """
        Iteratively edits the crystal graph by removing edges based on threshold weights
        until reaching 0-periodic substructures.

        Returns:
            defaultdict: A dictionary of restored substructures categorized by periodicity.

        The method performs the following steps:
        - Iterates over threshold weights and removes weaker interatomic contacts.
        - Stops when reaching 0-periodicity substructures at most.
        - Determines the lattice transformation matrix (LTM) if not already identified.
        - Restores intra-fragment contacts in low-periodic substructures.
        """

        monitor = []  # Tracks the progress of graph edits
        deleted_contacts = []  # Keeps track of removed interatomic contacts
        threshold_deleted_contacts = defaultdict(list)  # Stores contacts removed at each threshold
        low_periodic_substructures = []  # Stores intermediate substructures
        sg_copy = copy.copy(self.sg)

        for threshold_weight in self._get_threshold_weights():
            periodicity_before = get_dimensionality_larsen(sg_copy)
            edges_to_remove_at_threshold_weight = []

            # Identify edges that should be removed at this threshold
            for edge in sg_copy.graph.edges(data=True):

                node1, node2, edge_data = edge

                # Check bond property thresholds
                if (
                    (edge_data['weight'] < threshold_weight and self.bond_property in ('BV', 'SA', 'A', 'PI')) or
                    (edge_data['weight'] > threshold_weight and self.bond_property in ('R', ))
                ):
                    # Record the broken bond
                    broken_bond = sorted([sg_copy.graph.nodes[node]['element'] for node in (node1, node2)])
                    broken_bond_elements = tuple(broken_bond)

                    # ((1, 2, (1, 0, -1), edge_data dict, ['El1', 'El2'])
                    deleted_contacts.append(
                        ((node1, node2, edge_data['to_jimage']), edge_data, broken_bond)
                    )
                    threshold_deleted_contacts[(broken_bond_elements, edge_data['BV'])].append(
                        ((node1, node2, edge_data['to_jimage']), edge_data)
                    )
                    edges_to_remove_at_threshold_weight.append((node1, node2, edge_data['to_jimage']))

            # Remove identified edges at threshold t
            for edge_to_remove in edges_to_remove_at_threshold_weight:
                sg_copy.break_edge(*edge_to_remove)

            periodicity_after = get_dimensionality_larsen(sg_copy)
            bvs_after_editing = self._calculate_SG_BV_sum(sg_copy)
            monitor.append(
                (threshold_weight, '..'.join(broken_bond_elements),
                 periodicity_before, periodicity_after,
                 len(edges_to_remove_at_threshold_weight), bvs_after_editing)
            )

            # Store the structure graph at the step when a low-periodic (2-p, 1-p, 0-p) substructure appears
            if periodicity_before != periodicity_after:
                self._substructure_periodicities.add(periodicity_after)
                low_periodic_substructures.append(
                    {
                        'max_periodicity': periodicity_after,
                        'sg': copy.copy(sg_copy),
                        'deleted_contacts': copy.copy(threshold_deleted_contacts),  # NB!!! copy.copy(deleted_contacts)
                     }
                )

            # Stop if 0-periodic substructures are reached
            if periodicity_after == 0:
                self.monitor = monitor
                break

        # Identify lattice transformation required to obtain cell in which layers are in (001) rods along [001]
        self._determine_lattice_transformation(low_periodic_substructures)

        # Restore intra-fragment contacts and return the final substructures
        restored_crystal_substructures = self._restore_intra_contacts1(low_periodic_substructures)

        return restored_crystal_substructures

    def _analyze_graph2(self) -> Optional[TargetSubstructure]:
        """
        Iteratively edits the crystal graph by removing edges based on threshold weights
        until reaching the target periodicity.

        Returns:
            TargetSubstructure: The restored target substructure if found, otherwise None.

        The method performs the following steps:
        - Iterates over threshold weights and removes weaker interatomic contacts.
        - Tracks broken bonds and stores them in `deleted_contacts`.
        - Stops when reaching the target periodicity or when no further removal is possible.
        - Restores intra-fragment contacts in the target substructure.
        """

        threshold_deleted_contacts = defaultdict(list)  # keep track of contacts broken during graph editing iterations
        sg_copy = copy.copy(self.sg)
        total_BVS = self._calculate_SG_BV_sum(self.sg)

        for threshold_weight in self._get_threshold_weights():
            edges_to_remove_at_threshold_weight = []

            # Identify edges that should be removed at this threshold
            for edge in sg_copy.graph.edges(data=True):

                node1, node2, edge_data = edge

                # Check bond property thresholds
                if (
                    (edge_data['weight'] < threshold_weight and self.bond_property in ('BV', 'SA', 'A', 'PI')) or
                    (edge_data['weight'] > threshold_weight and self.bond_property in ('R', ))
                ):

                    broken_bond = sorted([sg_copy.graph.nodes[node]['element'] for node in (node1, node2)])
                    broken_bond_elements = tuple(broken_bond)

                    threshold_deleted_contacts[(broken_bond_elements, edge_data['BV'])].append(
                        ((node1, node2, edge_data['to_jimage']), edge_data, broken_bond)
                    )
                    edges_to_remove_at_threshold_weight.append((node1, node2, edge_data['to_jimage']))

            for edge_to_remove in edges_to_remove_at_threshold_weight:
                sg_copy.break_edge(*edge_to_remove)

            periodicity_after = get_dimensionality_larsen(sg_copy)

            # Stop if the target periodicity is reached
            if periodicity_after > self.target_periodicity:
                continue
            elif periodicity_after == self.target_periodicity:
                target_substructure = {
                    'total_BVS': total_BVS,
                    'sg': copy.copy(sg_copy),
                    'deleted_contacts': threshold_deleted_contacts,
                }
                break
            # If periodicity drops below the target, stop searching
            elif periodicity_after < self.target_periodicity:
                return None

        # Restore intra-fragment contacts and return the final substructure
        restored_target_substructure = self._restore_intra_contacts2(target_substructure)

        return restored_target_substructure

    def analyze_graph(self, N=2) -> CrystalSubstructures:
        """
        Performs a full analysis of the crystal graph by iteratively removing edges
        until the target periodicity substructure are identified.

        Returns:
            CrystalSubstructures: A structured object containing identified substructures.

        The method performs the following steps:
        - Runs the first structure graph analysis (`_analyze_graph1`) to identify lattice transformations.
        - Changes the lattice accordingly, re-calculates the connectivity
        - Runs the second graph analysis (`_analyze_graph2`) to extract the target periodic substructure.
        - Organizes the identified substructures into a `CrystalSubstructures` object.
        """

        # Step 1: Run the first graph analysis to determine lattice transformation matrix
        restored_crystal_substructures = self._analyze_graph1()

        # Step 2: Create a new structure graph using the identified lattice transformation
        self._create_structure_graph(transform_lattice=True, calculate_valences=True, N=N)

        # Step 3: Run the second graph analysis to extract the target substructure
        restored_target_crystal_substructure = self._analyze_graph2()

        # Step 4: Organize results into a structured CrystalSubstructures object
        crystal_substructures = CrystalSubstructures.from_dict(
            self,
            restored_crystal_substructures,
            restored_target_crystal_substructure
        )

        return crystal_substructures
