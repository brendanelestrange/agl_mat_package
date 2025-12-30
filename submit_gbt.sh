#!/bin/bash
#SBATCH -J gradboost  ## Name of job, you can define it to be any word
#SBATCH -A ISAAC-UTK0323  	##Information about the project account to be charged
#SBATCH --nodes=1              ## Number of nodes
#SBATCH --ntasks-per-node=2  ##-ntasks is used when we want to define total number of processors
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB 
#SBATCH --partition=ai-tenn     ## At present, we have four partitions. Check the different partitions using sinfo command
#SBATCH -e myjob.e%j           ## Errors will be written in this file 
#SBATCH -o myjob.o%j           ## output of the run will be written here
#SBATCH --qos=ai-tenn

#SBATCH --mail-user=brendan@vols.utk.edu
#SBATCH --mail-type=ALL

module load cuda/11.8.0-binary
module load anaconda3/2024.06 
source $ANACONDA3_SH
conda activate cool
~/.conda/envs/cool/bin/python model.py -f Features/tmQMg_properties_and_targets_AGL_TM_CPP_k112_c6.0.csv

