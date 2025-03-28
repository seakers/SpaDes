#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=012725TransferLearningTest.slurm
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --output=012725TransferLearningTest3Output.%j
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu

# modules needed for running DL jobs. Module restore will also work
module restore SpaDesCollection

# Python venv
# source activate_venv SpaDesVenv
source $SCRATCH/SpaDesVenv/bin/activate

# scripts or executables
cd $SCRATCH/SpaDes
# pip list

python ConfigurationDesign.py