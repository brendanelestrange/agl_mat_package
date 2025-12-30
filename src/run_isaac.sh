#!/bin/bash
#SBATCH -J agl_score_py  ## Name of job, you can define it to be any word
#SBATCH -A ISAAC-UTK0323  	##Information about the project account to be charged
#SBATCH --nodes=1              ## Number of nodes
#SBATCH --ntasks-per-node=2  ##-ntasks is used when we want to define total number of processors
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB 
#SBATCH --partition=ai-tenn    ## At present, we have four partitions. Check the different partitions using sinfo command
#SBATCH -e myjob.e%j           ## Errors will be written in this file 
#SBATCH -o myjob.o%j           ## output of the run will be written here
#SBATCH --qos=ai-tenn

#SBATCH --mail-user=brendan@vols.utk.edu
#SBATCH --mail-type=ALL

module load anaconda3/2024.06 
source $ANACONDA3_SH        
conda activate cool
python get_agl_mat_features.py -k 112 -c 6.0 -f ../data/tmQMg/tmQMg_properties_and_targets.csv -m Adjacency -dd ../data/tmQMg/xyz --f ../Features

mv myjob.* output_features