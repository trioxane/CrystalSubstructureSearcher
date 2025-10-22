#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import logging
import os

import pandas as pd
import ray
import yaml
import json
from pymatgen.analysis.local_env import VoronoiNN

from css import utils
from css.StructureAnalyzer import CrystalSubstructureSearcher
from css.structure_classes import CrystalSubstructureSearcherResults

warnings.filterwarnings('ignore')

# Suppress Ray logging
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for HPC cluster run configuration.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - folder: Path to folder with CIF files
            - num_cpus: Number of CPUs to use
            - timeout: Timeout per task in seconds
            - params: Path to parameters YAML file
    """
    parser = argparse.ArgumentParser(
        description="Ray-based parallel Crystal Substructure Search for HPC clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python CSS_HPC_run.py -f /path/to/cifs --num-cpus 64 --params params.yaml
  python CSS_HPC_run.py -f /path/to/cifs --num-cpus 128 --timeout 600
        """
    )

    parser.add_argument(
        '-f', '--folder',
        type=str,
        required=True,
        help='Path to the folder with CIF files'
    )

    parser.add_argument(
        '-n', '--num-cpus',
        type=int,
        default=16,
        help='Number of CPUs to use (default: 16)'
    )

    parser.add_argument(
        '-t', '--timeout',
        type=int,
        default=120,
        help='Maximum runtime per task in seconds (default: 120)'
    )

    parser.add_argument(
        '--max-runtime',
        type=int,
        default=(23 * 60 + 55) * 60,
        help='Maximum total runtime in minutes (default: 23h 55m). Job will save results and exit gracefully when time limit is reached.'
    )

    parser.add_argument(
        '-p', '--params',
        type=str,
        default='./params.yaml',
        help='Path to file with CSS calculation parameters (default: ./params.yaml)'
    )

    return parser.parse_args()


@ray.remote(num_cpus=1)
def process_single_cif(
        cif_file: str,
        config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run CrystalSubstructureSearcher calculation on a single CIF file.

    This function is executed as a Ray remote task, allowing parallel
    processing of multiple CIF files with dynamic load balancing.

    Args:
        cif_file: Path to CIF file to process
        config: Configuration dictionary containing parameters

    Returns:
        Dictionary containing analysis results or error information
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

    filename = Path(cif_file).stem
    tic = time.time()

    try:
        css = CrystalSubstructureSearcher(
            file_name=cif_file,
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

    return result


def main() -> None:
    """
    Main function managing Ray-based parallel CrystalSubstructureSearcher runs.

    Orchestrates the following workflow:
    1. Initializes Ray cluster with specified number of CPUs
    2. Validates input folder and parameters file
    3. Loads configuration from YAML
    4. Submits all CIF files as individual Ray tasks
    5. Collects results dynamically as tasks complete
    6. Combines all results into a single Excel output file
    7. Provides progress updates during execution
    """
    args = parse_arguments()

    # === Initialize Ray ===
    ray.init(num_cpus=args.num_cpus)
    print(f"\n{'=' * 70}")
    print(f"Ray initialized with {args.num_cpus} CPUs")
    print(f"{'=' * 70}\n")

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
    for key in ['save_low_p_components_path', 'save_slab_as_cif_path']:
        folder_path = config['paths'].get(key)
        if folder_path:
            Path(folder_path).mkdir(parents=True, exist_ok=True)

    # === Get all CIF files ===
    cif_files = glob.glob(str(input_folder / "*.cif"))

    total_files = len(cif_files)
    print(f"Found {total_files} CIF files to process")
    print(f"Timeout per task: {args.timeout} seconds\n")
    print(f"Maximum total runtime: {args.max_runtime:.1f} minutes")

    # === Put config in Ray object store (shared memory, not copied) ===
    config_ref = ray.put(config)

    # === Submit all tasks ===
    print("Submitting tasks...")
    futures = [process_single_cif.remote(cif_file, config_ref) for cif_file in cif_files]

    # === Collect results dynamically with progress tracking ===
    collected_results = []
    start_time = time.time()
    max_end_time = start_time + args.max_runtime * 60  # max-runtime in minutes

    print(f"{'=' * 70}")
    print("Processing files (dynamic load balancing)...")
    print(f"{'=' * 70}\n")

    while len(futures) > 0:

        # Check if we've exceeded maximum runtime
        if max_end_time and time.time() >= max_end_time:
            print(f"\n{'!' * 70}")
            print(f"!!! Maximum runtime reached")
            print(f"!!! Stopping gracefully and saving {len(collected_results)} completed results")
            print(f"!!! {len(futures)} tasks will be cancelled")
            print(f"{'!' * 70}\n")

            # Cancel remaining tasks
            for future in futures:
                ray.cancel(future)

            break

        # Wait for at least one task to complete (or timeout)
        ready_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=args.timeout)

        if ready_futures:
            # Get results from completed tasks
            results = ray.get(ready_futures)
            collected_results.extend(results)

            # Update progress
            completed = len(collected_results)
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_files - completed) / rate if rate > 0 else 0

            print(f"Progress: {completed}/{total_files} ({100 * completed / total_files:.1f}%) | "
                  f"Rate: {rate:.2f} files/sec | ETA: {eta / 60:.1f} min")

        futures = remaining_futures

    print(f"\n{'='*70}")
    if len(futures) == 0:
        print(f"✓ All {total_files} files processed successfully")
    else:
        print(f"!!! Completed {len(collected_results)}/{total_files} files before time limit")
        print(f"  {len(futures)} files were not processed")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"{'='*70}\n")

    # === Combine and save all results ===
    print("Saving results...")

    df_all_results = pd.DataFrame(collected_results).explode(
        column=['orientation_in_original_cell', 'component_formula', 'periodicity',
                'estimated_charge', 'inter_bvs_per_unit_size',
                'geometric_layer_thickness', 'physical_layer_thickness', 'save_path']
    )

    df_all_results.to_excel(f"CSS_HPC_results_{timestamp}.xlsx", index=False)

    # === Save raw results as JSON for backup ===
    json_output = f"CSS_HPC_results_{timestamp}.json"
    with open(json_output, "w") as f:
        json.dump(collected_results, f, cls=utils.npEncoder, indent=4)

    # === Shutdown Ray ===
    ray.shutdown()


if __name__ == '__main__':
    main()