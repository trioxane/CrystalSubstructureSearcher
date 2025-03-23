from collections import defaultdict, Counter
from typing import List, Dict, Union, Tuple, Set, Optional
from pathlib import Path

import networkx as nx
import pandas as pd
import numpy as np

from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.dimensionality import get_structure_components


import utils
from element_data import get_element_grouping_dict, ALVAREZ_VDW_RADIUS_DICT, MAX_VALENCE_DICT


class Contact:

    def __init__(
            self,
            pmg_structure: Structure,
            at1_idx: int,
            at2_idx: int,
            t: Tuple,
            contact_characteristics: Dict,
            element_grouping_dict: Optional[Dict] = None,
            identify_WP: bool = False,
    ):

        self.t: Tuple = t
        self.at1_idx: int = at1_idx
        self.at2_idx: int = at2_idx

        self._fill_attributes(pmg_structure, contact_characteristics, element_grouping_dict, identify_WP)

    def _fill_attributes(
            self, pmg_structure: Structure, contact_characteristics: Dict, element_grouping_dict: Dict, identify_WP: bool
    ) -> None:

        self.at1: str = pmg_structure[self.at1_idx].specie.symbol
        self.at2: str = pmg_structure[self.at2_idx].specie.symbol
        self.bond: str = '..'.join(sorted((self.at1, self.at2)))

        if element_grouping_dict is not None:
            self.at1_grouped: str = element_grouping_dict[self.at1]
            self.at2_grouped: str = element_grouping_dict[self.at2]
            self.bond_grouped: str = '..'.join(sorted((self.at1_grouped, self.at2_grouped)))
        else:
            self.bond_grouped = None

        self.BV: float = contact_characteristics['BV']
        self.BV_calc_method: str = contact_characteristics['BV_calc_method']
        self.R: float = contact_characteristics['R']
        self.A: float = contact_characteristics['A']
        self.SA: float = contact_characteristics['SA']
        self.PI: float = contact_characteristics['PI']

        self.at1_frac_coords: np.ndarray = pmg_structure.frac_coords[self.at1_idx]
        self.at2_frac_coords: np.ndarray = pmg_structure.frac_coords[self.at2_idx]

        contact_midpoint = (self.at2_frac_coords + self.t + self.at1_frac_coords) / 2
        # print(self.at1_frac_coords, self.t, self.at2_frac_coords)
        # print(contact_midpoint)
        self.contact_midpoint: np.ndarray = np.round(np.mod(contact_midpoint, 1.0), decimals=4)

        if identify_WP:
            self.contact_midpoint_WP: str = utils.determine_wyckoff_position(contact_midpoint, struct=pmg_structure)
        else:
            self.contact_midpoint_WP = ''

    @classmethod
    def from_data(cls, pmg_structure, at1_idx, at2_idx, t, contact_characteristics, element_grouping_dict, identify_WP):
        return cls(pmg_structure, at1_idx, at2_idx, t, contact_characteristics, element_grouping_dict, identify_WP)


class Substructure:

    def __init__(
            self,
            max_periodicity: int,
            sg: StructureGraph,
            deleted_contacts: Optional[defaultdict],
            inter_contacts: Set,
            BVS: float,
    ):

        self.max_periodicity = max_periodicity
        self.sg = sg
        self.deleted_contacts = deleted_contacts
        self.inter_contacts = inter_contacts
        self.BVS = BVS

    def iter_components(self, inc_orientation=True, inc_site_ids=True) -> Dict:
        component_iterator = get_structure_components(
            self.sg, inc_orientation=inc_orientation, inc_site_ids=inc_site_ids
        )
        return component_iterator

    def __len__(self) -> int:
        return len(tuple(self.iter_components()))


class TargetSubstructure(Substructure):

    def __init__(
            self,
            max_periodicity: int,
            sg: StructureGraph,
            deleted_contacts: defaultdict,
            inter_contacts: Set,
            intercomponent_contacts_in_original_cell: list[Contact],
            BVS: float,
            total_BVS: float,
            ltm_used: np.ndarray,
    ):

        super().__init__(max_periodicity, sg, deleted_contacts, inter_contacts, BVS)

        self.total_BVS = total_BVS
        self.ltm_used = ltm_used
        self.intercomponent_contacts_in_original_cell = intercomponent_contacts_in_original_cell

        self.component_charges: Dict[int, float] = None
        self.component_graphs: List[StructureGraph] = None
        self.component_orientations: List[Union[int, Tuple[int]]] = None
        self.component_periodicity: List[int] = None

        self.components_per_cell_count: int = None
        self.components_per_cell_formula: str = None
        self.component_formulas: List[str] = None
        self.unique_component_indices: List[int] = None

        self.component_dimensions: Dict[int, Dict] = dict()

        self._calculate_component_data()
        self._get_unique_component_indices()
        self._calculate_component_charges()

    def _calculate_component_data(self) -> None:

        components_data = list(self.iter_components(inc_orientation=True, inc_site_ids=False))

        self.component_graphs = [component_data['structure_graph'] for component_data in components_data]
        self.component_orientations = [component_data['orientation'] for component_data in components_data]
        self.component_periodicity = [component_data['dimensionality'] for component_data in components_data]

        self.component_formulas = [sg.structure.composition.formula.replace(" ", "") for sg in self.component_graphs]
        self.components_per_cell_count = len(components_data)
        self.components_per_cell_formula = '_'.join(
            [f"({specie}){counts}" for specie, counts in Counter(self.component_formulas).items()]
        )

    def _get_unique_component_indices(self) -> None:
        """
        Identifies unique graphs in a list of component graphs and returns the indices
        of the last occurrence of each unique graph.

        Returns:
            list: Indices of the last occurrence of each unique graph.
        """
        # DONE check maybe store last appearances not the first ones
        # check graphs isomorphism by comparing node elements and edge distances
        node_matcher = nx.algorithms.isomorphism.categorical_node_match(attr='element', default='')
        edge_matcher = nx.algorithms.isomorphism.numerical_multiedge_match(attr='R', default=1.0, atol=1e-1)

        unique_component_indices = []
        for i in range(len(self.component_graphs) - 1, -1, -1):  # Iterate in reverse
            is_unique = True
            for j in range(len(self.component_graphs) - 1, i, -1):  # Check later indices first
                if nx.is_isomorphic(
                        self.component_graphs[i].graph,
                        self.component_graphs[j].graph,
                        node_match=node_matcher,
                        edge_match=edge_matcher,
                ):
                    is_unique = False
                    break
            if is_unique:
                unique_component_indices.append(i)

        unique_component_indices = sorted(unique_component_indices)

        print(f'unique_component_indices: {unique_component_indices}')
        self.unique_component_indices = unique_component_indices

        return None

    def _calculate_component_charges(self) -> None:
        """
        Estimate component charge using the EN difference of the atoms in the interfragment contact.
        Atom with higher EN belonging to the component contributes the partial negative charge to that
        component amounting to contact BV value; atom with lower EN contributes the partial positive
        charge to that component, respectively.
        """

        EPSILON = 1e-6
        component_charges = defaultdict(float)

        component_site_dict = {}
        for idx, component in enumerate(self.iter_components(inc_orientation=False, inc_site_ids=True)):
            component_site_dict[idx] = component['site_ids']

        site_component_dict = {
            site: component
            for component, sites in component_site_dict.items()
            for site in sites
        }

        node_electronegativities = {i: self.sg.graph.nodes[i]['EN'] for i in range(self.sg.graph.number_of_nodes())}
        contacts_bv = {
            contact_data[0]: contact_data[1]['BV']
            for removed_bonds_set in self.deleted_contacts.values()
            for contact_data in removed_bonds_set
        }

        for inter_contact in self.inter_contacts:
            node1, node2, _ = inter_contact
            contact_bv = contacts_bv[inter_contact]
            en_difference = node_electronegativities[node1] - node_electronegativities[node2]
            # print(sg.structure[node1].specie.symbol, sg.structure[node2].specie.symbol,
            #       inter_contact, contact_bv, en_difference)
            if en_difference > EPSILON:
                component_charges[site_component_dict[node1]] -= contact_bv
                component_charges[site_component_dict[node2]] += contact_bv
            elif en_difference < -EPSILON:
                component_charges[site_component_dict[node1]] += contact_bv
                component_charges[site_component_dict[node2]] -= contact_bv
            else:
                component_charges[site_component_dict[node1]] += 0.0
                component_charges[site_component_dict[node2]] += 0.0

        self.component_charges = {idx: round(chg, 4) for idx, chg in sorted(component_charges.items())}

        print(f'component_charges: {self.component_charges}')

        return None


class CrystalSubstructures:

    def __init__(self, css_instance):

        self.monitor = css_instance.monitor
        self.crystal_graph_name = css_instance.crystal_graph_name
        self.bond_property = css_instance.bond_property
        self.suspicious_contacts = css_instance.suspicious_contacts
        self.target_periodicity = css_instance.target_periodicity

        self._ltm_applied = css_instance._ltm

        self.atom_data = css_instance.sg.graph.nodes(data=True)
        self._BVS_x_periodicity: Optional[Dict] = None
        self._BVS_x_periodicity_normalized: Optional[Dict] = None

        self.target_substructure: Optional[TargetSubstructure] = None
        self.substructures: defaultdict[int, List[Substructure]] = defaultdict()

        self._calculate_BVS_x_periodicity()

    @classmethod
    def from_dict(
            cls, css_instance, substructures_dict: defaultdict, target_substructure: Union[None, TargetSubstructure]
    ):

        cs = cls(css_instance=css_instance)
        cs.substructures = substructures_dict
        cs.target_substructure = target_substructure
        cs._calculate_BVS_x_periodicity()

        return cs

    def iter_substructures(self):
        return self.substructures.items()

    @property
    def show_monitor(self):
        df = pd.DataFrame(self.monitor, columns=[
            f'threshold_{self.bond_property}', 'broken_bond', 'periodicity_before',
            f'periodicity_after', 'N_edges_removed', 'bvs_after_editing'
        ])

        return df

    def _calculate_BVS_x_periodicity(self) -> None:

        self._BVS_x_periodicity = {int(p): substr[0].BVS for p, substr in self.iter_substructures()}

        partition_abs = np.abs(np.ediff1d(list(self._BVS_x_periodicity.values())))
        partition_diff = np.ediff1d(list(self._BVS_x_periodicity.values()))

        print('partition "abs"', partition_abs)
        print('partition "diff"', partition_diff)

        # structures in which BVS N-p < BVS (N-1)-p are unusual and this partition is not defined for them
        if any(diff > 0 for diff in partition_diff):
            self._BVS_x_periodicity_normalized = {'3-p': np.nan, '2-p': np.nan, '1-p': np.nan}
        else:
            self._BVS_x_periodicity_normalized = {
                f"{p}-p": x for p, x in zip(self._BVS_x_periodicity.keys(), partition_abs / partition_abs.sum())
            }

        return None

    @property
    def show_BVS_x_periodicity_normalized(self):
        return pd.Series(
            name=f"BVS_x_periodicity_normalized for {self.crystal_graph_name}",
            data=self._BVS_x_periodicity_normalized.values(),
            index=self._BVS_x_periodicity_normalized.keys()
        )

    def save_substructure_components(
            self,
            save_components_path: str = './',
            save_substructure_as_cif: bool = True,
            save_bulk_as_cif: bool = False,
            store_symmetrized_cell: bool = False,
            vacuum_space: float = 20.0,
    ) -> None:

        def _transform_layer_cell():

            a_vec, b_vec, c_vec = component_sg.structure.lattice.matrix

            ab_normal = np.cross(a_vec, b_vec)
            ab_normal /= np.linalg.norm(ab_normal)
            new_lattice = [a_vec, b_vec, ab_normal * vacuum_space]

            layer_with_vacuum = Structure(
                new_lattice,
                component_sg.structure.species_and_occu,
                component_sg.structure.cart_coords,
                site_properties=component_sg.structure.site_properties,
                coords_are_cartesian=True,
            )
            # layer_with_vacuum.to('check_layer_w_vacuum.cif')

            # get dimensions of the layer
            top_atom_idx = layer_with_vacuum.frac_coords[:, 2].argmax()
            bottom_atom_idx = layer_with_vacuum.frac_coords[:, 2].argmin()
            # print(layer_with_vacuum[top_atom_idx].specie.symbol)
            top_atom_radius = ALVAREZ_VDW_RADIUS_DICT[layer_with_vacuum[top_atom_idx].specie.symbol]
            bottom_atom_radius = ALVAREZ_VDW_RADIUS_DICT[layer_with_vacuum[bottom_atom_idx].specie.symbol]
            geometric_layer_thickness = layer_with_vacuum.lattice.c * (
                    layer_with_vacuum.frac_coords[:, 2].max() - layer_with_vacuum.frac_coords[:, 2].min()
            )
            physical_layer_thickness = geometric_layer_thickness + top_atom_radius + bottom_atom_radius
            print(f"geometrical layer thickness: {geometric_layer_thickness:.2f} Angstrom")
            print(f"physical layer thickness: {physical_layer_thickness:.2f} Angstrom")

            new_lattice2 = [a_vec, b_vec, ab_normal * (vacuum_space + geometric_layer_thickness)]

            layer_w_vacuum2 = Structure(
                new_lattice2,
                layer_with_vacuum.species_and_occu,
                layer_with_vacuum.cart_coords,
                site_properties=layer_with_vacuum.site_properties,
                coords_are_cartesian=True,
            )

            # layer_w_vacuum2.to('check_layer_w_vacuum2.cif')
            S1 = utils.calculate_area(layer_w_vacuum2.lattice.matrix)

            sga = SpacegroupAnalyzer(layer_w_vacuum2)
            layer_w_vacuum_symmetrised = sga.get_conventional_standard_structure()
            # layer_w_vacuum_symmetrised.to('check_layer_w_vacuum_symmetrised.cif')
            S2 = utils.calculate_area(layer_w_vacuum_symmetrised.lattice.matrix)

            # TODO skip at the moment standardizing because in some cases (ICSD 653676 or 262075)
            #  layer in the standardized cell is not in (001) plane
            layer_w_vacuum_symmetrised = layer_w_vacuum2

            if (
                    layer_w_vacuum_symmetrised.frac_coords[:, 2].max() -
                    layer_w_vacuum_symmetrised.frac_coords[:, 2].min()
            ) > 1 - 4/vacuum_space:

                print('layer is divided')
                layer_w_vacuum_symmetrised.translate_sites(
                    list(range(len(layer_w_vacuum_symmetrised))),
                    (0.0, 0.0, 0.5),
                    to_unit_cell=True,
                )
            else:
                print('layer is inside')
                c_shift = layer_w_vacuum_symmetrised.frac_coords.mean(axis=0)[2]
                print('c_shift:', c_shift)
                layer_w_vacuum_symmetrised.translate_sites(
                    list(range(len(layer_w_vacuum_symmetrised))),
                    (0.0, 0.0, -c_shift + 0.5),
                    to_unit_cell=True,
                )

            print(f"S1: {S1:.1f}, S2: {S2:.1f}")

            return S1, geometric_layer_thickness, physical_layer_thickness, layer_w_vacuum_symmetrised

        if self.target_substructure is not None:

            if save_bulk_as_cif:
                self.target_substructure.sg.structure.to(
                    Path(save_components_path) / f"{self.crystal_graph_name}_bulk_transformed.cif", fmt='cif'
                )

            print('unique_component_indices:', self.target_substructure.unique_component_indices)
            for idx, component in enumerate(
                    self.target_substructure.iter_components(inc_orientation=True, inc_site_ids=False)
            ):
                component_sg = component['structure_graph']
                periodicity = component['dimensionality']
                component_composition = component_sg.structure.composition.formula.replace(" ", "")
                print('save_substructure_components:', idx, periodicity, component['orientation'], component_composition)

                if (
                        (idx in self.target_substructure.unique_component_indices) and
                        (periodicity == self.target_periodicity) and
                        (self.target_periodicity == 2)  # for the time being only layer saving is available
                ):

                    S, geometric_layer_thickness, physical_layer_thickness = None, None, None

                    save_path = (Path(save_components_path) /
                                 f"{self.crystal_graph_name}-{self.bond_property}-"
                                 f"component_{idx}-{component_composition}-{periodicity}p.cif")

                    # if layer is already completely inside the cell store right away the structure
                    if set([d[2][2] for d in component_sg.graph.edges(data='to_jimage')]) == {0}:
                        print('layer is wholly inside unit cell')

                        if store_symmetrized_cell:
                            S, geometric_layer_thickness, physical_layer_thickness, component_structure = _transform_layer_cell()
                        else:
                            component_structure = component_sg.structure

                        if save_substructure_as_cif:
                            component_structure.to(save_path, fmt='cif')

                    # otherwise we need to first put the layer inside
                    else:
                        print('layer is divided by unit cell borders')

                        # make supercell; the bond translation vectors "to_jimage" will be recalculated
                        g_new = component_sg * [1, 1, 2]

                        restoration_succeeded = False
                        for _, component in enumerate(
                                get_structure_components(g_new, inc_orientation=True, inc_site_ids=False)
                        ):
                            p, component_sg = component['dimensionality'], component['structure_graph']

                            # TODO fix this after pymatgen update
                            if p == 3 or set([d[2][2] for d in component_sg.graph.edges(data='to_jimage')]) == {0}:

                                if store_symmetrized_cell:
                                    S, geometric_layer_thickness, physical_layer_thickness, component_structure = _transform_layer_cell()
                                else:
                                    component_structure = component_sg.structure

                                restoration_succeeded = True

                                if save_substructure_as_cif:
                                    component_structure.to(save_path, fmt='cif')

                        if not restoration_succeeded:
                            raise utils.StructureGraphAnalysisException("pymatgen error")

                    self.target_substructure.component_dimensions[idx] = {
                        'S': S,
                        'geometric_layer_thickness': geometric_layer_thickness,
                        'physical_layer_thickness': physical_layer_thickness,
                        'save_path': str(save_path.absolute()) if save_substructure_as_cif else '',
                    }


class CrystalSubstructureSearcherResults:
    """
    A class to store the results of crystal graph analysis.

    Args:
        crystal_substructures (CrystalSubstructures): An instance of CrystalSubstructures.

    Attributes:
        target_periodicity_reached (bool): Whether the target periodicity was reached.
        input_file_name (str): Name of the input file.
        bond_property (str): The selected bond property.
        monitor (pd.DataFrame): DataFrame with monitoring data stored at each crystal graph editing step.
        total_bvs (float): Total bond valence sum.
        fragments (dict): Dictionary of fragments.
        xbvs (float): Fraction of bond valence sum in the fragments.
        mean_inter_bv (float): Mean bond valence of contacts between fragments.
        inter_bvs_per_interface (float): Bond valence sum per interface.
        interfragment_contact_atoms (dict): Atom types involved in inter-fragment contacts.
        interfragment_contact_arbitrary_types (dict): Arbitrary atom types involved in inter-fragment contacts;
            elements are grouped according to self.arbitrary_types_dict (utils.py).
        inter_contacts_bv_estimate (dict): Estimated bond valence for inter-fragment contacts.
        intra_bvs (float): Intra-fragment bond valence sum.
        inter_bvs (float): Inter-fragment bond valence sum.
        edited_graph_total_bvs (float): Total bond valence sum of the edited graph.
        restored_graph_total_bvs (float): Total bond valence sum of the restored graph.
    """

    def __init__(
            self,
            crystal_substructures: CrystalSubstructures,
            element_grouping: Union[int, Dict] = 1,
    ):
        """
        Initialize the CrystalSubstructureSearcherResults.

        Args:
            crystal_substructures (CrystalSubstructures): An instance of CrystalSubstructures.
        """
        self.crystal_graph_name = crystal_substructures.crystal_graph_name
        self.suspicious_contacts = crystal_substructures.suspicious_contacts
        self.bond_property = crystal_substructures.bond_property
        self.target_periodicity = crystal_substructures.target_periodicity
        self._monitor = crystal_substructures.monitor

        self.BVS_x_periodicity = crystal_substructures._BVS_x_periodicity
        self.BVS_x_periodicity_normalized = crystal_substructures._BVS_x_periodicity_normalized

        self._calculate_atom_valence_data(crystal_substructures.atom_data)

        if crystal_substructures.target_substructure is not None:

            self.structure = crystal_substructures.target_substructure.sg.structure
            self.ltm_used = crystal_substructures.target_substructure.ltm_used
            self.intra_bvs = crystal_substructures.target_substructure.BVS
            self.total_bvs = crystal_substructures.target_substructure.total_BVS
            self.unique_component_indices = crystal_substructures.target_substructure.unique_component_indices
            self.component_charges = crystal_substructures.target_substructure.component_charges
            self.component_formulas = crystal_substructures.target_substructure.component_formulas
            self.component_dimensions = crystal_substructures.target_substructure.component_dimensions
            self.component_orientations = crystal_substructures.target_substructure.component_orientations
            self.component_periodicity = crystal_substructures.target_substructure.component_periodicity
            self.components_per_cell_count = crystal_substructures.target_substructure.components_per_cell_count
            self.components_per_cell_formula = crystal_substructures.target_substructure.components_per_cell_formula

            self._inter_contacts = crystal_substructures.target_substructure.inter_contacts
            self._deleted_contacts = crystal_substructures.target_substructure.deleted_contacts
            self.intercomponent_contacts_in_original_cell = crystal_substructures.target_substructure.intercomponent_contacts_in_original_cell

            self._element_grouping_dict = get_element_grouping_dict(grouping=element_grouping)

            self._calculate_additional_results()

        else:
            self.intra_bvs = np.nan
            self.total_bvs = self.BVS_x_periodicity[3]
            self.unique_component_indices = ()
            self.component_charges = dict()
            self.component_formulas = ()
            self.component_dimensions = dict()
            self.component_orientations = ()
            self.component_periodicity = ()
            self.components_per_cell_count = 0
            self.components_per_cell_formula = ''

            self._inter_contacts = set()
            self._deleted_contacts = dict()

            self.inter_bvs = np.nan
            self.xbvs = np.nan
            self.intercomponent_contact_atoms = dict()
            self.intercomponent_contact_atoms_groups = dict()
            self.intercomponent_contacts_bv_estimate = dict()
            self.intercomponent_contacts_area = dict()
            self.intercomponent_contacts_in_original_cell = set()

            self.mean_inter_bv = np.nan
            self.inter_bvs_per_interface = np.nan
            self.contact_A_weighted_inter_bv = np.nan

            self.intercomponent_contact_midpoint_WP = ''

    @property
    def show_monitor(self):
        df = pd.DataFrame(self._monitor, columns=[
            f'threshold_{self.bond_property}', 'broken_bond', 'periodicity_before',
            f'periodicity_after', 'N_edges_removed', 'bvs_after_editing'
        ])

        return df

    @property
    def show_BVS_x_periodicity_normalized(self):
        return pd.Series(
            name=f"BVS_x_periodicity_normalized for {self.crystal_graph_name}",
            data=self.BVS_x_periodicity_normalized.values(),
            index=self.BVS_x_periodicity_normalized.keys()
        )

    def _get_original_substructure_orientations(self, transformed_cell_orientation: Tuple) -> Union[str, Tuple]:
        if transformed_cell_orientation is not None:
            return tuple(np.linalg.inv(self.ltm_used).dot(transformed_cell_orientation).astype(int))
        else:
            return ''

    def _calculate_atom_valence_data(self, node_view: nx.Graph.nodes) -> None:
        """
        Calculate attributes related to the atom valences in the structure.

        atom_valences (Dict): dictionary with keys element symbols
            and values the set of unique valences for the element in the crystal structure
        suspicious_valences (Set): set storing the strings with element, site_index and atom valence information

        Args:
            node_view: data for nodes of the structure graph
        """

        atom_valences = defaultdict(set)
        suspicious_valences = set()

        for i, n_dict in node_view:
            element, valence = n_dict['element'], n_dict['valence']
            if valence > MAX_VALENCE_DICT[element]:
                suspicious_valences.add(f'{element}{i} {valence:.1f} vu')
            atom_valences[element].add(valence)

        self.atom_valences = {k: sorted(v) for k, v in atom_valences.items()}
        self.suspicious_valences = suspicious_valences

        return None

    def _calculate_additional_results(self) -> None:

        self.inter_bvs = self.total_bvs - self.intra_bvs
        self.xbvs = self.intra_bvs / self.total_bvs
        self.inter_bvs_per_interface = self.inter_bvs / self.components_per_cell_count  # TODO: check this division

        # these data correspond to the transformed cell
        intercomponent_contacts = []
        # just a reminder that dictionary self.deleted_contacts is
        # ((at1_string, at2_string), BV_float): List[Tuple[Tuple[at1_idx_int, at2_idx_int, Tuple[translation]], Dict]
        for contact_type, contacts_list in self._deleted_contacts.items():
            for contact in contacts_list:
                if contact[0] in self._inter_contacts:

                    intercomponent_contacts.append(
                        Contact.from_data(
                            pmg_structure=self.structure,
                            at1_idx=contact[0][0],
                            at2_idx=contact[0][1],
                            t=contact[0][2],
                            contact_characteristics=contact[1],
                            element_grouping_dict=self._element_grouping_dict,
                            identify_WP=False
                        )
                    )

        # self.intercomponent_contacts = intercomponent_contacts
        self.mean_inter_bv = self.inter_bvs / len(intercomponent_contacts)
        BV_A = [(c.BV, c.A) for c in intercomponent_contacts]
        self.contact_A_weighted_inter_bv = utils.get_weighted_mean(BV_A)
        BV_count = [(c.BV, 1) for c in intercomponent_contacts]
        self.weighted_inter_bv = utils.get_weighted_mean(BV_count)

        dd = defaultdict(float)
        for contact in intercomponent_contacts:
            dd[contact.bond] += contact.A
        self.intercomponent_contacts_area = {k: round(v, 2) for k, v in dd.items()}

        self.intercomponent_contact_atoms = pd.Series(
            [c.bond for c in intercomponent_contacts]).value_counts().to_dict()
        self.intercomponent_contact_atoms_groups = pd.Series(
            [c.bond_grouped for c in intercomponent_contacts]).value_counts().to_dict()
        self.intercomponent_contacts_bv_estimate = pd.Series(
            [c.BV_calc_method for c in intercomponent_contacts]
        ).value_counts(normalize=True).apply(lambda v: round(v, 4)).to_dict()

        # these data correspond to the original cell
        if getattr(self.intercomponent_contacts_in_original_cell.copy().pop(), 'contact_midpoint_WP', False):
            self.intercomponent_contact_midpoint_WP = sorted(
                {(c.bond, round(c.R, 3), c.contact_midpoint_WP)
                 for c in self.intercomponent_contacts_in_original_cell},
                key=lambda v: v[0],
            )

        return None

    def as_string(self) -> str:
        """
        Get the results as a formatted string.

        Returns:
            dict: Dictionary containing formatted results.
        """
        results_dict = {attribute: getattr(self, attribute) for attribute in vars(self) if not attribute.startswith('_')}

        formatted_string = f"\n{'#' * 50}\nRESULTS\n{'#' * 50}\n"
        for key, value in results_dict.items():
            formatted_string += f"{key}: {value}\n"
        formatted_string += '#' * 50

        return ''.join(formatted_string)

    def as_dict(self) -> Dict:
        """
        Get the results as a dictionary.

        Returns:
            dict: Dictionary containing results.
        """
        results = {
            'crystal_graph_name': self.crystal_graph_name,
            'bond_property': self.bond_property,
            'target_periodicity': self.target_periodicity,
            # 'LTM': tuple(tuple(arr) for arr in self.ltm_used),
            'orientation_in_original_cell': [self._get_original_substructure_orientations(self.component_orientations[idx])
                                             for idx in self.unique_component_indices],
            'composition': [self.component_formulas[idx] for idx in self.unique_component_indices],
            'periodicity': [self.component_periodicity[idx] for idx in self.unique_component_indices],
            'estimated_charge': [self.component_charges[idx] for idx in self.unique_component_indices],
            # 'S_A^2': [self.component_dimensions.get(idx, dict()).get('S', np.nan)
            #           for idx in self.unique_component_indices],
            'geometric_layer_thickness': [self.component_dimensions.get(idx, dict()).get('geometric_layer_thickness', np.nan)
                                          for idx in self.unique_component_indices],
            'physical_layer_thickness': [self.component_dimensions.get(idx, dict()).get('physical_layer_thickness', np.nan)
                                          for idx in self.unique_component_indices],
            'save_path': [self.component_dimensions.get(idx, dict()).get('save_path', '')
                          for idx in self.unique_component_indices],
            'components_per_cell_count': self.components_per_cell_count,
            'components_per_cell_formula': self.components_per_cell_formula,

            'BVS_x_periodicity': self.BVS_x_periodicity,
            'xbvs': self.xbvs,
            'mean_inter_bv': self.mean_inter_bv,
            'mean_A_weighted_inter_bv': self.contact_A_weighted_inter_bv,
            'inter_bvs_per_interface': self.inter_bvs_per_interface,
            'inter_bvs_per_unit_area': [
                self.inter_bvs_per_interface / self.component_dimensions.get(idx, dict()).get('S', np.nan)
                for idx in self.unique_component_indices
            ],
            'inter_contact_atoms': '|'.join(sorted(self.intercomponent_contact_atoms.keys())),
            'inter_contact_atoms_area': self.intercomponent_contacts_area,
            'inter_contact_atoms_count': self.intercomponent_contact_atoms,
            'inter_contact_arbitrary_types_count': self.intercomponent_contact_atoms_groups,
            'inter_contact_arbitrary_types': '|'.join(sorted(self.intercomponent_contact_atoms_groups.keys())),
            'inter_contacts_bv_ML_estimated_(unreliable_share)': self.intercomponent_contacts_bv_estimate.get('ML_estimated (confidence: False)', 0.0),
            'intercomponent_contact_midpoint_WP': self.intercomponent_contact_midpoint_WP,

            'suspicious_contacts': sorted(self.suspicious_contacts),
            'suspicious_valences': sorted(self.suspicious_valences),
            'atom_valences': self.atom_valences,
        }

        results.update(self.BVS_x_periodicity_normalized)

        return results
