#!/bin/bash
set -e

conda env create -f environment.yml
conda activate rmpx-hashing-benchmark
MLFLOW_TRACKING_URI=http://acireale.iiar.pwr.edu.pl/mlflow/ mlflow run . -P trials=10000 -P rmpx-size=11 -P hash=md5 -P agent=yacs -P modulo=8

echo "OK"
