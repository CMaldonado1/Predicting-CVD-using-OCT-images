# Predicting-CVD-using-OCT-images
This reposotory contain the code for the paper "Predicting risk of cardiovascular disease using retinal optical coherence tomography imaging""

## Docker Image

The Docker file used for this work can be found at the following link:

[Docker image](https://hub.docker.com/layers/scclmgadmin/oct/oct/images/sha256-af579dc2cab9c7504937fdea208c683a61603126fbab4ccf641a4c2bef71b043?context=repo)

## Overview

In the `src/` folder, you'll find the scripts needed to train the Variational Autoencoder, specifically with this command:

python oct_main_ml_fc2.py

You can modify the arguments to conduct a hyperparameter search grid.

An example of a bash script to run the hyperparameter script is also included: `hyperparameter_grid_search.sh`.
In this bash script provides an example of running 4 jobs in parallel simultaneously for all grid searches; the feasibility of this will depend on the capacity of the HPC you are using.
In the `jupyter-notebooks/` folder, you'll find the notebooks for the seven classifiers tasks. Make sure to include the Excel files that will contain the Z's and metadata information.

In the jupyter-notebooks are the code of the Random Forest classifiers (the seven specifies in the manuscript)



