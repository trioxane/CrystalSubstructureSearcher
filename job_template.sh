#!/bin/bash
#SBATCH --job-name=css_run_part{{PARTITION_ID}}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000MB
#SBATCH -A IscrB_ONEDER
#SBATCH --time=23:59:00
#SBATCH -p dcgp_usr_prod
#SBATCH --output={{WORK_DIR}}/css_partition_{{PARTITION_ID}}_%j.out


# Activate virtual environment
source /leonardo/home/userexternal/pzolotar/CrystalSubstructureSearcher_HPC/venv/bin/activate

# Change to partition directory
cd {{PARTITION_DIR}}

# Run CSS script
python3 {{SCRIPT_PATH}} \
    --folder {{PARTITION_DIR}} \
    --num-cpus 1 \
    --timeout 120 \
    --max-runtime 1435 \
    --params {{WORK_DIR}}/params.yaml

exit $?
