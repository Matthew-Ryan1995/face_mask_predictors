#!/bin/bash
#SBATCH -N 1               	                                # number of nodes (no MPI, so we only use a single node)
#SBATCH -n 30            	                                # number of cores
#SBATCH --time=02:00:00    	                                # walltime allocation, which has the format (D-HH:MM:SS), here set to 1 hour
#SBATCH --mem=10GB         	                                # memory required per node (here set to 4 GB)
#SBATCH --output=slurm_outputs/feature_importance/slurm-%A_%a.out


# Notification configuration
#SBATCH --array=1-12
#SBATCH --mail-type=END					    	# Send a notification email when the job is done (=END)


#loading modules
module load python/3.12.3 

source $(which virtualenvwrapper_lazy.sh)

workon face-mask-predictors

# Execute the program

python3 code/9_permute_feature_importance.py
