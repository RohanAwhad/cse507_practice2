#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -p general
#SBATCH -c 8            # number of cores 
#SBATCH -t 0-09:00:00   # time in d-hh:mm:ss
#SBATCH --mem=32G
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
module load mamba/latest
source activate practice_2_env

python /home/rawhad/CSE507/cse507_practice3/preprocessor.py
