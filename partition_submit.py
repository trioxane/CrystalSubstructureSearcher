#!/usr/bin/env python3
# coding: utf-8

"""
Partition CIF files and submit separate SLURM jobs for each partition.
"""

import argparse
import glob
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Partition CIF files and submit SLURM jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-f', '--folder',
        type=str,
        required=True,
        help='Path to folder containing CIF files'
    )

    parser.add_argument(
        '-N', '--num-partitions',
        type=int,
        default=4,
        help='Number of partitions (default: 4)'
    )

    parser.add_argument(
        '-s', '--script',
        type=str,
        default='./CSS_HPC_run.py',
        help='Path to python script to run (default: ./CSS_HPC_run.py)'
    )

    parser.add_argument(
        '-p', '--params',
        type=str,
        default='./params.yaml',
        help='Path to params.yaml file (default: ./params.yaml)'
    )

    parser.add_argument(
        '-v', '--venv',
        type=str,
        default='./venv/bin/activate',
        help='Path to virtual environment activate script (default: ./venv/bin/activate)'
    )

    parser.add_argument(
        '-t', '--template',
        type=str,
        default='job_template.sh',
        help='SLURM job template file (default: job_template.sh)'
    )

    return parser.parse_args()


def partition_files(cif_files: List[str], num_partitions: int, work_dir: Path) -> List[Path]:
    """
    Distribute CIF files across N partition folders.

    Args:
        cif_files: List of CIF file paths
        num_partitions: Number of partitions to create
        work_dir: Working directory for partitions

    Returns:
        List of partition directory paths
    """
    partition_dirs = []

    # Create partition folders
    for i in range(num_partitions):
        partition_dir = work_dir / f"partition_{i}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        partition_dirs.append(partition_dir)

    # Distribute files
    for idx, cif_file in enumerate(cif_files):
        partition_idx = idx % num_partitions
        shutil.copy2(cif_file, partition_dirs[partition_idx])

    return partition_dirs


def create_job_script(
        template_path: Path,
        partition_dir: Path,
        partition_id: int,
        script_path: Path,
        params_path: Path,
        venv_activate: Path,
        work_dir: Path,
) -> Path:
    """
    Create SLURM job script from template.

    Args:
        template_path: Path to job template file
        partition_dir: Partition directory
        partition_id: Partition index
        script_path: Path to Python script
        params_path: Path to params.yaml
        venv_activate: Path to venv activate script
        work_dir: Working directory

    Returns:
        Path to created job script
    """
    with open(template_path, 'r') as f:
        template = f.read()

    # Replace placeholders
    job_script_content = template.replace('{{PARTITION_ID}}', str(partition_id))
    job_script_content = job_script_content.replace('{{PARTITION_DIR}}', str(partition_dir))
    job_script_content = job_script_content.replace('{{SCRIPT_PATH}}', str(script_path))
    job_script_content = job_script_content.replace('{{PARAMS_PATH}}', str(params_path))
    job_script_content = job_script_content.replace('{{VENV_ACTIVATE_PATH}}', str(venv_activate))
    job_script_content = job_script_content.replace('{{WORK_DIR}}', str(work_dir))

    # Write job script
    job_script_path = partition_dir / 'submit_job.sh'
    with open(job_script_path, 'w') as f:
        f.write(job_script_content)

    os.chmod(job_script_path, 0o755)

    return job_script_path


def submit_job(job_script: Path) -> str:
    """
    Submit job to SLURM queue.

    Args:
        job_script: Path to job script

    Returns:
        Job ID as string
    """
    result = subprocess.run(
        ['sbatch', str(job_script)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit job: {result.stderr}")

    # Extract job ID from output: "Submitted batch job 12345"
    job_id = result.stdout.strip().split()[-1]
    return job_id


def main() -> None:
    """Main function."""
    args = parse_arguments()

    # Validate inputs
    input_folder = Path(args.folder).resolve()
    script_path = Path(args.script).resolve()
    params_path = Path(args.params).resolve()
    venv_activate = Path(args.venv).resolve()
    template_file = Path(args.template).resolve()
    script_dir = Path.cwd()

    if not input_folder.exists():
        print(f"Error: Input folder not found: {input_folder}")
        return

    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return

    if not params_path.exists():
        print(f"Error: Parameters file not found: {params_path}")
        return

    if not venv_activate.exists():
        print(f"Error: Virtual environment activate script not found: {venv_activate}")
        return

    if not template_file.exists():
        print(f"Error: Template file not found: {template_file}")
        return

    # Get CIF files
    cif_files = glob.glob(str(input_folder / "*.cif"))

    if not cif_files:
        print(f"Error: No CIF files found in {input_folder}")
        return

    print(f"Found {len(cif_files)} CIF files")

    # Create work directory
    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M')
    work_dir = script_dir / f"partitions_{timestamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Work directory: {work_dir}")

    # Partition files
    print(f"Creating {args.num_partitions} partitions...")
    partition_dirs = partition_files(cif_files, args.num_partitions, work_dir)

    # Print distribution
    for i, partition_dir in enumerate(partition_dirs):
        num_files = len(list(partition_dir.glob("*.cif")))
        print(f"  Partition {i}: {num_files} files")

    # Create and submit jobs
    print("Submitting jobs...")
    job_ids = []

    for i, partition_dir in enumerate(partition_dirs):
        job_script = create_job_script(
            template_file,
            partition_dir,
            i,
            script_path,
            params_path,
            venv_activate,
            work_dir,
        )

        job_id = submit_job(job_script)
        job_ids.append(job_id)

        print(f"  Partition {i}: Job {job_id}")
        time.sleep(1.0)

    print(f"\nSubmitted {len(job_ids)} jobs: {' '.join(job_ids)}")
    print(f"Results in: {work_dir}/partition_*/")


if __name__ == '__main__':
    main()