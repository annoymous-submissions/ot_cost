#!/bin/bash


datasets=("Credit" "Weather")
hyperparameter="False"
personal="False"

for dataset in "${datasets[@]}"; do
    echo "Submitting job for $dataset"
    sbatch script_run_models $dataset $hyperparameter $personal
done


datasets=("EMNIST" "CIFAR" "IXITiny" "ISIC")
datasets=("IXITiny" "ISIC")
hyperparameter="False"
personal="False"

for dataset in "${datasets[@]}"; do
    echo "Submitting job for $dataset"
    sbatch script_gpu_run_models $dataset $hyperparameter $personal
done
