#!/usr/bin/env python
# coding: utf-8
import warnings
import glob
import time
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from pymatgen.analysis.local_env import VoronoiNN

import utils
from StructureAnalyzer import CrystalSubstructureSearcher
from structure_classes import CrystalSubstructureSearcherResults


warnings.filterwarnings('ignore')


def main():

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

    run_results = []

    for i, f in enumerate(glob.glob(r'C:/Users/pavel.zolotarev/Desktop/icsd_BVS/binding_energies_all/*.cif')):
    # for f in glob.glob(r'C:/Users/pavel.zolotarev/Desktop/icsd_BVS/binding_energies_all/*_form2.cif'):
    # for i, f in enumerate(list(glob.glob(r'C:/Users/pavel.zolotarev/Desktop/icsd_BVS/binding_energies_all/*.cif'))):

        if i == 200:
            break

        filename = Path(f).stem
        print(f'Start analysis: {filename}')
        tic = time.time()

        try:
            css = CrystalSubstructureSearcher(
                file_name=f,
                connectivity_calculator=connectivity_calculator,
                bond_property=BOND_PROPERTY,
                target_periodicity=TARGET_PERIODICITY,
            )
            crystal_substructures = css.analyze_graph()

            crystal_substructures.save_substructure_components(
                save_components_path='./test_save_cif',
                save_substructure_as_cif=True,
                save_bulk_as_cif=False,
                store_symmetrized_cell=True,
                vacuum_space=20.0,
            )

            results = CrystalSubstructureSearcherResults(
                crystal_substructures=crystal_substructures,
                element_grouping=1,
            )
            result = results.as_dict()

        except Exception as e:
            result = {'crystal_graph_name': filename, 'error': e.__class__.__name__, 'error_message': str(e)}
        finally:
            result.update({'runtime_sec': round(time.time() - tic, 2)})
            run_results.append(result)

    pd.DataFrame(run_results)\
        .explode(column=['orientation_in_original_cell', 'composition', 'periodicity', 'estimated_charge',
                         'inter_bvs_per_unit_area', 'geometric_layer_thickness', 'physical_layer_thickness', 'save_path'])\
        .to_excel(f"CSS_Results_{datetime.today().strftime('%d-%m-%Y_%H-%M')}.xlsx")

    with open(f"CSS_Results_{datetime.today().strftime('%d-%m-%Y_%H-%M')}.json", "w") as out_json:
        json.dump(run_results, out_json, cls=utils.npEncoder)


if __name__ == '__main__':
    main()
