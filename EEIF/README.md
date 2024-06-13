# EEiF: Efficient Isolated Forest with $e$ Branches for Anomaly Detection
This repository contains the code for the experiments of the paper "EEiF: Efficient Isolated Forest with $e$ Branches for Anomaly Detection".

# Requirement
- numpy==1.20.1
- sklearn==0.22.1
- pandas==1.4.1

# Dataset
We evaluate all methods on 15 widely-used benchmark datasets

# Experiment
### You can try different `--threshold` (the cut threshold) to see how the AUC performance changes. 
### An example for running the code:
    python demo.py --threshold=403 
