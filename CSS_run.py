#!/usr/bin/env python
# coding: utf-8

import warnings
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from pymatgen.analysis.local_env import VoronoiNN

from css import utils
from css.StructureAnalyzer import CrystalSubstructureSearcher
from css.structure_classes import CrystalSubstructureSearcherResults

warnings.filterwarnings('ignore')


# === Default parameters dictionary ===
# option not available for change are marked with # sign
DEFAULT_PARAMS = {
    'connectivity_calculation_parameters': {
        'tol': 0.1,
        'cutoff': 6.0,
        'extra_nn_info': True,  #
        'allow_pathological': True,  #
        'compute_adj_neighbors': False,  #
    },
    'structure_graph_analysis_parameters': {
        'bond_property': 'BV',
        'target_periodicity': 2,
        'supercell_extent': 2,
    },
    'save_options': {
        'save_substructure_as_cif': True,  #
        'save_structure_with_bond_midpoints': False,  #
        'save_slab_as_cif': False,
        'save_bulk_as_cif': False,  #
        'store_symmetrized_cell': True,  #
        'vacuum_space': 20.0,
        'min_slab_thickness': 20.0,
    },
    'element_grouping': {
        'element_grouping_dict': 1,
    }
}


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run CSS on a single CIF file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python CSS_run.py -f input.cif
  python CSS_run.py -f input.cif -o ./my_results
  python CSS_run.py -f input.cif --bond-property SA --target-periodicity 1
  python CSS_run.py -f input.cif --save-slab --min-slab-thickness 25.0 --vacuum-space 25.0
  python CSS_run.py -f input.cif --supercell-extent 3
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        required=True,
        help='Path to the input CIF file'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: ./CSS_results_{input_file_stem})'
    )
    
    parser.add_argument(
        '--bond-property',
        type=str,
        default=None,
        choices=['BV', 'R', 'SA', 'A', 'PI'],
        help='Bond property to use as a bond strength criteria. Options: BV, R, SA, A, PI (default: BV)'
    )
    
    parser.add_argument(
        '--target-periodicity',
        type=int,
        default=None,
        choices=[1, 2],
        help='Target periodicity (default: 2)'
    )
    
    parser.add_argument(
        '--supercell-extent',
        type=int,
        default=None,
        help='Supercell extent (default: 2)'
    )
    
    parser.add_argument(
        '--cutoff',
        type=float,
        default=None,
        help='cutoff parameter for VoronoiNN (default: 6.0)'
    )
    
    parser.add_argument(
        '--tol',
        type=float,
        default=None,
        help='tol parameter for VoronoiNN (default: 0.1)'
    )
    
    parser.add_argument(
        '--save-slab',
        action='store_true',
        help='Save slab as CIF (available only for 2-p substructures)'
    )
    
    parser.add_argument(
        '--vacuum-space',
        type=float,
        default=None,
        help='Vacuum space for saved slabs and single fragments (default: 20.0 angstrom)'
    )

    parser.add_argument(
        '--min-slab-thickness',
        type=float,
        default=None,
        help='Minimum slab thickness (default: 20.0 angstrom)'
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # === Validate input file ===
    cif_file = Path(args.file)
    if not cif_file.exists():
        raise FileNotFoundError(f"Input CIF file not found: {cif_file}")

    # === Set output directory default based on input file stem ===
    if args.output_dir is None:
        args.output_dir = f"./CSS_results_{cif_file.stem}"

    print(f"Input file: {cif_file}")

    # === Load default parameters ===
    conn_cfg = DEFAULT_PARAMS['connectivity_calculation_parameters'].copy()
    graph_cfg = DEFAULT_PARAMS['structure_graph_analysis_parameters'].copy()
    save_cfg = DEFAULT_PARAMS['save_options'].copy()
    grouping_cfg = DEFAULT_PARAMS['element_grouping'].copy()

    # === Override with command line arguments ===
    if args.bond_property:
        graph_cfg['bond_property'] = args.bond_property

    if args.target_periodicity is not None:
        graph_cfg['target_periodicity'] = args.target_periodicity

    if args.supercell_extent is not None:
        graph_cfg['supercell_extent'] = args.supercell_extent

    if args.cutoff is not None:
        conn_cfg['cutoff'] = args.cutoff

    if args.tol is not None:
        conn_cfg['tol'] = args.tol

    if args.save_slab:
        if graph_cfg['target_periodicity'] == 2:
            save_cfg['save_slab_as_cif'] = True
        else:
            raise RuntimeError('save_slab_as_cif option is available only if target_periodicity == 2')

    if args.vacuum_space is not None:
        save_cfg['vacuum_space'] = args.vacuum_space

    if args.min_slab_thickness is not None:
        save_cfg['min_slab_thickness'] = args.min_slab_thickness

    # === Create output directories ===
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Bond property: {graph_cfg['bond_property']}")
    print(f"Target periodicity: {graph_cfg['target_periodicity']}")
    print(f"Supercell extent: {graph_cfg['supercell_extent']}")
    print(f"Cutoff: {conn_cfg['cutoff']}, Tolerance: {conn_cfg['tol']}")

    # === Timestamp for consistent naming ===
    timestamp = datetime.today().strftime('%d-%m-%Y_%H-%M-%S')
    filename = cif_file.stem

    # === Connectivity calculator ===
    connectivity_calculator = VoronoiNN(
        tol=conn_cfg['tol'],
        cutoff=conn_cfg['cutoff'],
        extra_nn_info=conn_cfg['extra_nn_info'],
        allow_pathological=conn_cfg['allow_pathological'],
        compute_adj_neighbors=conn_cfg['compute_adj_neighbors'],
    )

    try:
        print(f"\nProcessing: {filename}")

        # === Run analysis ===
        css = CrystalSubstructureSearcher(
            file_name=str(cif_file),
            connectivity_calculator=connectivity_calculator,
            bond_property=graph_cfg['bond_property'],
            target_periodicity=graph_cfg['target_periodicity'],
        )

        tic = time.time()  # timer start
        crystal_substructures = css.analyze_graph(N=graph_cfg['supercell_extent'])

        print("Analysis completed successfully")

        # === Save results ===
        crystal_substructures.save_substructure_components(
            save_components_path=str(output_dir),
            save_substructure_as_cif=save_cfg['save_substructure_as_cif'],
            save_structure_with_bond_midpoints=save_cfg['save_structure_with_bond_midpoints'],
            save_slab_as_cif=save_cfg['save_slab_as_cif'],
            save_slab_as_cif_path=str(output_dir),
            min_slab_thickness=save_cfg['min_slab_thickness'],
            save_bulk_as_cif=save_cfg['save_bulk_as_cif'],
            store_symmetrized_cell=save_cfg['store_symmetrized_cell'],
            vacuum_space=save_cfg['vacuum_space'],
        )

        # === Generate results dictionary ===
        results = CrystalSubstructureSearcherResults(
            crystal_substructures=crystal_substructures,
            element_grouping=grouping_cfg['element_grouping_dict'],
        )
        result = results.as_dict()
        result['crystal_graph_name'] = filename
        result['runtime_sec'] = round(time.time() - tic, 2)

        # === Save output files ===
        output_json = output_dir / f"CSS_results_{filename}_{timestamp}.json"
        with open(output_json, "w") as f:
            json.dump(result, f, cls=utils.npEncoder, indent=2)

        print(f"Results saved to: {output_json}")

        print(f"\n✓ Successfully processed: {filename}")
        print(f"  Output directory: {output_dir}")
        print(f"  Runtime: {result['runtime_sec']} seconds")

    except Exception as e:
        error_msg = f"{e.__class__.__name__}: {str(e)}"
        print(f"\n✗ Error processing {filename}:")
        print(f"  {error_msg}")

        error_result = {
            'crystal_graph_name': filename,
            'error': e.__class__.__name__,
            'error_message': str(e),
            'runtime_sec': round(time.time() - tic, 2)
        }

        output_json = output_dir / f"CSS_erroneous_results_{filename}_{timestamp}.json"
        with open(output_json, "w") as f:
            json.dump(error_result, f, indent=2)

        raise


if __name__ == '__main__':
    main()
