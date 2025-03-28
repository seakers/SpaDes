#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=TestJob
#SBATCH --time=00:30:00
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --output=TestOutput.%j
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# modules needed for running DL jobs. Module restore will also work
module restore SpaDesCollection

# Python venv
source $SCRATCH/SpaDesVenv/bin/activate

# scripts or executables
cd $SCRATCH/SpaDes

python ConfigurationDesign.py