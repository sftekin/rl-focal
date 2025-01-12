#!/bin/bash
#SBATCH -J rl-async-mmlu                   # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=120G
#SBATCH -o Report-rl-async-mmlu.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=stekin6@gatech.edu        # E-mail address for notifications
cd  $HOME/scratch/rl-focal                            # Change to working directory
conda activate llamas
chmod +x run_mmlu.sh
./run_mmlu.sh