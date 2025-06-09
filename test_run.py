#!/usr/bin/env python
# coding: utf-8
from pymatgen.analysis.local_env import VoronoiNN
from StructureAnalyzer import CrystalSubstructureSearcher
from structure_classes import CrystalSubstructureSearcherResults

from warnings import filterwarnings
filterwarnings('ignore')

def main():

    # Example Usage:
    CIF_FILE_NAME = r'./154_Ca2Sb.cif'
    BOND_PROPERTY = 'BV'  # Choose the bond property to be used as a crystal graph editing criteria (BV, R, SA, A, PI)
    TARGET_PERIODICITY = 2  # Save the substructures with target periodicity (1 or 2)

    # bond determination class
    # larger tol means less interatomic contacts taken into account
    connectivity_calculator = VoronoiNN(
        tol=0.1,
        cutoff=6.0,
        extra_nn_info=True,
        allow_pathological=True,
        compute_adj_neighbors=False,
    )

    css = CrystalSubstructureSearcher(
        file_name=CIF_FILE_NAME,
        connectivity_calculator=connectivity_calculator,
        bond_property=BOND_PROPERTY,
        target_periodicity=TARGET_PERIODICITY,
    )
    crystal_substructures = css.analyze_graph()
    print(crystal_substructures.show_monitor)
    print(crystal_substructures._BVS_x_periodicity)
    print(crystal_substructures.show_BVS_x_periodicity_normalized)

    crystal_substructures.save_substructure_components(
        save_components_path='./',
        save_substructure_as_cif=True,
        save_structure_with_bond_midpoints=True,
        save_structure_with_bond_midpoints_path='./',
        save_slab_as_cif=True,
        save_slab_as_cif_path='./',
        save_bulk_as_cif=True,
        store_symmetrized_cell=True,
        vacuum_space=20.0,
    )

    results = CrystalSubstructureSearcherResults(
        crystal_substructures=crystal_substructures,
        element_grouping=1,
    )
    print(results.as_dict())
    print('finished')


if __name__ == '__main__':
    main()
