#!/bin/bash
#SBATCH -J musr                  # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=120G
#SBATCH -o Report-%J.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=stekin6@gatech.edu        # E-mail address for notifications
cd  $HOME/scratch/rl-focal                            # Change to working directory
conda init
conda activate llamas
python run.py --task_name musr --update_freq 10 --select_agent_epoch 25 --ensemble_agent_epoch 150 --ens_agent_lr 0.0001 --alpha 0
python run.py --task_name musr --update_freq 10 --select_agent_epoch 25 --ensemble_agent_epoch 150 --ens_agent_lr 0.0001 --alpha 0
python run.py --task_name musr --update_freq 10 --select_agent_epoch 25 --ensemble_agent_epoch 150 --ens_agent_lr 0.0001 --alpha 0

