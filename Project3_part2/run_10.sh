# === SETTINGS ===
project_dir="/zhome/91/d/188852/DP/project_3/samme_part1"

# === JOB SCRIPT ===
#!/bin/sh
#BSUB -q c02516
#BSUB -J run_model
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -W 10:00
#BSUB -o run_model_10_%J.out
#BSUB -e run_model_10_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate late_fusion

cd $project_dir

python -u weak.py --generate_clicks --train --pos 10 --neg 10

