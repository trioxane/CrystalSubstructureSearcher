#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import glob
import itertools
import time
import warnings
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from pymatgen.analysis.local_env import VoronoiNN

from css.StructureAnalyzer import CrystalSubstructureSearcher
from css.structure_classes import CrystalSubstructureSearcherResults

warnings.filterwarnings('ignore')


def parse_arguments():
    """
    Parse command line arguments for cluster run configuration.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - folder: Path to folder with CIF files
            - parallelisation: Number of processes to spawn
            - params: Path to parameters YAML file
    """
    parser = argparse.ArgumentParser(
        description="Parallelized Crystal Substructure Search runs across multiple cores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python CSS_parallel_run.py -N 2 -f /path/to/cifs -p params.yaml
        """
    )

    parser.add_argument(
        '-f', '--folder',
        type=str,
        required=True,
        help='Path to the folder with CIF files'
    )

    parser.add_argument(
        '-N', '--parallelisation',
        type=int,
        default=4,
        help='Number of processes to spawn (default: 4)'
    )

    parser.add_argument(
        '-p', '--params',
        type=str,
        default='./params.yaml',
        help='Path to file with CSS calculation parameters (default: ./params.yaml)'
    )

    return parser.parse_args()


def partition_input_folder(input_folder: str, num_partitions: int) -> List[List[str]]:
    """
    Partition CIF files from input folder into N groups for parallel processing.

    Args:
        input_folder: Path to folder containing CIF files
        num_partitions: Number of partitions to create

    Returns:
        List of lists, where each sublist contains paths to CIF files for one partition

    Raises:
        FileNotFoundError: If no CIF files are found in the input folder
    """
    cif_files = glob.glob(str(Path(input_folder) / "*.cif"))

    if not cif_files:
        raise FileNotFoundError(f"No CIF files found in {input_folder}")

    # Partition files into N groups
    partitions = [[] for _ in range(num_partitions)]

    for i, cif_path in enumerate(cif_files):
        partition_idx = i % num_partitions
        partitions[partition_idx].append(cif_path)

    return partitions


def css_partition_run(
        cif_file_list: List[str],
        config: Dict[str, Any],
        partition_id: int
) -> List[Dict[str, Any]]:
    """
    Run CrystalSubstructureSearcher calculation on a separate partition.

    Args:
        cif_file_list: List of paths to CIF files to process
        config: Configuration dictionary containing parameters
        partition_id: ID of the current partition (for output naming)

    Returns:
        List of dictionaries containing results from each processed CIF file
    """
    paths = config['paths']
    conn_cfg = config['connectivity_calculation_parameters']
    graph_cfg = config['structure_graph_analysis_parameters']
    save_cfg = config['save_options']
    grouping_cfg = config['element_grouping']

    # === Connectivity calculator ===
    connectivity_calculator = VoronoiNN(
        tol=conn_cfg['tol'],
        cutoff=conn_cfg['cutoff'],
        extra_nn_info=conn_cfg['extra_nn_info'],
        allow_pathological=conn_cfg['allow_pathological'],
        compute_adj_neighbors=conn_cfg['compute_adj_neighbors'],
    )

    run_results = []

    for f in cif_file_list:
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

    # === Save partition results ===
    output_df = pd.DataFrame(run_results).explode(
        column=['orientation_in_original_cell', 'component_formula', 'periodicity',
                'estimated_charge', 'inter_bvs_per_unit_size',
                'geometric_layer_thickness', 'physical_layer_thickness', 'save_path']
    )

    output_df.to_excel(f"_CSS_results_partition_{partition_id}.xlsx")

    print(f"✓ Partition {partition_id} completed: {len(cif_file_list)} files processed")

    return run_results


def main() -> None:
    """
    Main function managing parallel CrystalSubstructureSearcher runs.

    Functions manages the following workflow:
    1. Reads configuration from YAML file
    2. Validates input folder and parameters file
    3. Partitions CIF files across N processes
    4. Distributes processing across multiple cores
    5. Collects results from all partitions
    6. Combines results into a single output Excel file
    """
    args = parse_arguments()

    # === Timestamp for consistent naming ===
    timestamp = datetime.today().strftime('%d-%m-%Y_%H-%M-%S')

    # === Validate input folder ===
    input_folder = Path(args.folder)
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # === Load parameters from YAML ===
    params_file = Path(args.params)
    if not params_file.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_file}")

    with open(params_file, "r") as f:
        config = yaml.safe_load(f)

    # === Create output folders if they don't exist ===
    for key in ['save_low_p_components_path', 'save_structure_with_bond_midpoints_path', 'save_slab_as_cif_path']:
        folder_path = config['paths'].get(key)
        if folder_path:
            Path(folder_path).mkdir(parents=True, exist_ok=True)

    # === Partition input folder into N parts ===
    partitions = partition_input_folder(str(input_folder), args.parallelisation)

    total_files = sum(len(p) for p in partitions)
    print(f"\n{'=' * 60}")
    print(f"Processing {total_files} CIF files across {args.parallelisation} processes")
    print(f"{'=' * 60}\n")

    # === Run calculations in parallel ===
    collected_run_results = []
    with Pool(args.parallelisation) as pool:
        partition_run_results = pool.starmap(
            css_partition_run,
            zip(partitions, itertools.repeat(config), range(args.parallelisation))
        )
        # Flatten list of lists into single list
        for partition_results in partition_run_results:
            collected_run_results.extend(partition_results)

    print(f"\n{'=' * 60}")
    print(f"✓ All {args.parallelisation} partitions completed")
    print(f"{'=' * 60}\n")

    # === Combine and save all results ===
    df_all_results = pd.DataFrame(collected_run_results).explode(
        column=['orientation_in_original_cell', 'component_formula', 'periodicity',
                'estimated_charge', 'inter_bvs_per_unit_size',
                'geometric_layer_thickness', 'physical_layer_thickness', 'save_path']
    )

    df_all_results.to_excel(f"CSS_parallel_results_{timestamp}.xlsx")

    # === Clean up temporary partition result files ===
    print("\nCleaning up temporary partition files...")
    for partition_id in range(args.parallelisation):
        temp_file = f"_CSS_results_partition_{partition_id}.xlsx"
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == '__main__':
    main()
