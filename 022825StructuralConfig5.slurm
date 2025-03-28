#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=022825StructuralConfig5.slurm
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --output=022825StructuralConfigOutput50Comp32b200e10r.%j
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu

# modules needed for running DL jobs. Module restore will also work
module restore SpaDesCollection

# Python venv
# source activate_venv SpaDesVenv
source $SCRATCH/virtual_envs/SpaDesEnv/bin/activate
export PIP_TARGET=$SCRATCH/.local/lib
export PYTHONPATH=$PIP_TARGET:$PYTHONPATH
export MPLCONFIGDIR=$SCRATCH/.local/lib/matplotlib-config

# scripts or executables
cd $SCRATCH/SpaDes

python -c "print('About to run ConfigurationDesignStruct.py')"

python ConfigurationDesignStruct.py

deactivate