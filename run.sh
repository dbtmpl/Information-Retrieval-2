#!/bin/bash

#SBATCH --job-name=example
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time 10:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module purge

#module load 2020
#module load pre2019
#module load Python/3.6.3-foss-2017b

module load 2019
module load Java
module load CUDA/10.0.130
module load Miniconda3

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

source activate InfRet2

srun python ../src/start.py ../src/config/ranker/conv_knrm_qqa ../src/config/qulac
