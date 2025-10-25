#!/usr/bin/env python3
# coding: utf-8

"""
Collect and merge results from partitioned CSS calculations
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect and merge results from partitioned calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_results.py -r partitions_20241023_1430 --cif-folder low_p_components
  python collect_results.py -r partitions_20241023_1430 --cif-folder slabs --no-merge-jsons
        """
    )

    parser.add_argument(
        '-r', '--result-folder',
        type=str,
        required=True,
        help='Path to folder containing partition_* subdirectories'
    )

    parser.add_argument(
        '--cif-folder',
        type=str,
        required=True,
        choices=['low_p_components', 'slabs'],
        help='Target folder name to extract CIF files from'
    )

    parser.add_argument(
        '--merge-xlsx',
        action='store_true',
        default=True,
        help='Merge Excel files (default: True)'
    )

    parser.add_argument(
        '--no-merge-xlsx',
        action='store_false',
        dest='merge_xlsx',
        help='Skip merging Excel files'
    )

    parser.add_argument(
        '--merge-jsons',
        action='store_true',
        default=True,
        help='Merge JSON files (default: True)'
    )

    parser.add_argument(
        '--no-merge-jsons',
        action='store_false',
        dest='merge_jsons',
        help='Skip merging JSON files'
    )

    return parser.parse_args()


def collect_cif_files(result_folder: Path, cif_folder_name: str, output_folder: Path) -> int:
    """
    Collect CIF files from partition subdirectories.

    Args:
        result_folder: Root folder containing partitions
        cif_folder_name: Name of subfolder containing CIFs (e.g., 'low_p_components')
        output_folder: Destination folder for collected CIFs

    Returns:
        Number of CIF files collected
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    partition_dirs = sorted(result_folder.glob("partition_*"))
    total_files = 0

    for partition_dir in partition_dirs:
        cif_source_dir = partition_dir / cif_folder_name

        if not cif_source_dir.exists():
            continue

        cif_files = list(cif_source_dir.glob("*.cif"))

        for cif_file in cif_files:
            # Create unique name: partition_N_original_name.cif
            partition_name = partition_dir.name
            new_name = f"{partition_name}_{cif_file.name}"
            dest_path = output_folder / new_name

            shutil.copy2(cif_file, dest_path)
            total_files += 1

    return total_files


def merge_excel_files(result_folder: Path, output_file: Path) -> int:
    """
    Merge Excel files from partition subdirectories.

    Args:
        result_folder: Root folder containing partitions
        output_file: Output Excel file path

    Returns:
        Number of Excel files merged
    """
    partition_dirs = sorted(result_folder.glob("partition_*"))
    dataframes = []
    files_found = 0

    for partition_dir in partition_dirs:
        excel_files = list(partition_dir.glob("*.xlsx"))

        for excel_file in excel_files:
            try:
                df = pd.read_excel(excel_file)
                # Add partition info column
                df.insert(0, 'partition', partition_dir.name)
                dataframes.append(df)
                files_found += 1
            except Exception as e:
                print(f"  Warning: Failed to read {excel_file}: {e}")

    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df.to_excel(output_file)
        return files_found
    else:
        print("  Warning: No Excel files found to merge")
        return 0


def merge_json_files(result_folder: Path, output_file: Path) -> int:
    """
    Merge JSON files from partition subdirectories.

    Args:
        result_folder: Root folder containing partitions
        output_file: Output JSON file path

    Returns:
        Number of JSON files merged
    """
    partition_dirs = sorted(result_folder.glob("partition_*"))
    merged_data = []
    files_found = 0

    for partition_dir in partition_dirs:
        json_files = list(partition_dir.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Handle both list and single object
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            item['partition'] = partition_dir.name
                        merged_data.append(item)
                else:
                    if isinstance(data, dict):
                        data['partition'] = partition_dir.name
                    merged_data.append(data)

                files_found += 1
            except Exception as e:
                print(f"  Warning: Failed to read {json_file}: {e}")

    if merged_data:
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=4)
        return files_found
    else:
        print("  Warning: No JSON files found to merge")
        return 0


def main() -> None:
    """Main function."""
    args = parse_arguments()

    # Validate result folder
    result_folder = Path(args.result_folder).resolve()

    if not result_folder.exists():
        print(f"Error: Result folder not found: {result_folder}")
        return

    # Check for partition subdirectories
    partition_dirs = list(result_folder.glob("partition_*"))

    if not partition_dirs:
        print(f"Error: No partition_* subdirectories found in {result_folder}")
        return

    print(f"Found {len(partition_dirs)} partition directories")

    # Output folder/file names
    result_folder_name = result_folder.name
    cif_output_folder = result_folder.parent / f"{result_folder_name}_cifs_{args.cif_folder}"
    xlsx_output_file = result_folder.parent / f"{result_folder_name}.xlsx"
    json_output_file = result_folder.parent / f"{result_folder_name}.json"

    # Collect CIF files
    num_cifs = collect_cif_files(result_folder, args.cif_folder, cif_output_folder)
    print(f"\nCollected {num_cifs} CIF files → {cif_output_folder}")

    # Merge Excel files
    if args.merge_xlsx:
        num_xlsx = merge_excel_files(result_folder, xlsx_output_file)
        print(f"\nMerged {num_xlsx} Excel files → {xlsx_output_file}")
    else:
        print("\nSkipping Excel merge (--no-merge-xlsx)")

    # Merge JSON files
    if args.merge_jsons:
        num_jsons = merge_json_files(result_folder, json_output_file)
        print(f"\nMerged {num_jsons} JSON files → {json_output_file}")
    else:
        print("\nSkipping JSON merge (--no-merge-jsons)")

    print("\nCollection complete!")


if __name__ == '__main__':
    main()