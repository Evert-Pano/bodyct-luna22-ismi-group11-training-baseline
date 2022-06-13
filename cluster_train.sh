#!/bin/sh

#Setting basic SBATCH properties
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=9
#SBATCH --gpus=8
#SBATCH --mem=128G
#SBATCH -t 0:10:00

#Loading some required modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
module load cuDNN/8.2.1.32-CUDA-11.3.1

#Install required dependancies
pip install --user numpy
pip install --user matplotlib
pip install --user tensorflow
pip install --user SimpleITK

#Call the training script
python3 $HOME/bodyct-luna22-ismi-group11-training-baseline/train.py
