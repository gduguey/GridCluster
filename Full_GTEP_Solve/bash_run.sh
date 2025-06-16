#!/bin/bash

source /etc/profile
module load anaconda/Python-ML-2024a
module load gurobi/gurobi-1102

python full_gtep_solve.py > log.txt
