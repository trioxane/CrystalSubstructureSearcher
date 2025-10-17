#!/usr/bin/env python
# coding: utf-8

import warnings
import glob
import time
import json
import yaml
from pathlib import Path
from datetime import datetime

import pandas as pd
from pymatgen.analysis.local_env import VoronoiNN

import utils
from StructureAnalyzer import CrystalSubstructureSearcher
from structure_classes import CrystalSubstructureSearcherResults

warnings.filterwarnings('ignore')


def main():
    # === Load parameters from YAML ===
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    paths = config['paths']
    conn_cfg = config['connectivity_calculation_parameters']
    graph_cfg = config['structure_graph_analysis_parameters']
    save_cfg = config['save_options']
    grouping_cfg = config['element_grouping']

    # === Create output folders if they don't exist ===
    for key in ['save_low_p_components_path', 'save_structure_with_bond_midpoints_path', 'save_slab_as_cif_path']:
        folder_path = paths.get(key)
        if folder_path:
            Path(folder_path).mkdir(parents=True, exist_ok=True)

    # === Timestamp for consistent naming ===
    timestamp = datetime.today().strftime('%d-%m-%Y_%H-%M-%S')

    # === Connectivity calculator ===
    connectivity_calculator = VoronoiNN(
        tol=conn_cfg['tol'],
        cutoff=conn_cfg['cutoff'],
        extra_nn_info=conn_cfg['extra_nn_info'],
        allow_pathological=conn_cfg['allow_pathological'],
        compute_adj_neighbors=conn_cfg['compute_adj_neighbors'],
    )

    run_results = []
    cif_files = glob.glob(str(Path(paths['input_cif_files_folder']) / "*.cif"))

    for _, f in enumerate(cif_files):
        filename = Path(f).stem
        tic = time.time()

        try:
            css = CrystalSubstructureSearcher(
                file_name=f,
                connectivity_calculator=connectivity_calculator,
                bond_property=graph_cfg['bond_property'],
                target_periodicity=graph_cfg['target_periodicity'],
            )
            crystal_substructures = css.analyze_graph(N=graph_cfg['supercell_extent'])

            crystal_substructures.save_substructure_components(
                save_components_path=paths['save_low_p_components_path'],
                save_substructure_as_cif=save_cfg['save_substructure_as_cif'],
                save_structure_with_bond_midpoints=save_cfg['save_structure_with_bond_midpoints'],
                save_structure_with_bond_midpoints_path=paths['save_structure_with_bond_midpoints_path'],
                save_slab_as_cif=save_cfg['save_slab_as_cif'],
                save_slab_as_cif_path=paths['save_slab_as_cif_path'],
                min_slab_thickness=save_cfg['min_slab_thickness'],
                save_bulk_as_cif=save_cfg['save_bulk_as_cif'],
                store_symmetrized_cell=save_cfg['store_symmetrized_cell'],
                vacuum_space=save_cfg['vacuum_space'],
            )

            results = CrystalSubstructureSearcherResults(
                crystal_substructures=crystal_substructures,
                element_grouping=grouping_cfg['element_grouping_dict'],
            )
            result = results.as_dict()

        except Exception as e:
            result = {
                'crystal_graph_name': filename,
                'error': e.__class__.__name__,
                'error_message': str(e)
            }
        finally:
            result.update({'runtime_sec': round(time.time() - tic, 2)})
            run_results.append(result)

    # === Save results ===
    pd.DataFrame(run_results).explode(
        column=['orientation_in_original_cell', 'component_formula', 'periodicity',
                'estimated_charge', 'inter_bvs_per_unit_size',
                'geometric_layer_thickness', 'physical_layer_thickness', 'save_path']
    ).to_excel(f"CSS_results_{timestamp}.xlsx")

    with open(f"CSS_results_{timestamp}.json", "w") as out_json:
        json.dump(run_results, out_json, cls=utils.npEncoder)

    with open(f"CSS_params_{timestamp}.yaml", "w") as out_yaml:
        yaml.dump(config, out_yaml, sort_keys=False)


if __name__ == '__main__':
    main()
