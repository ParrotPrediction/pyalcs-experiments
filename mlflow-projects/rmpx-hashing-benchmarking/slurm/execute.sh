#!/bin/bash
set -e

install_mlflow () {
pip install --target="$SLURM_SUBMIT_DIR/libs" mlflow
}

echo "Installing MLFlow"
install_mlflow

echo "Preparing conda environment"
export PYTHONPATH="$PYTHONPATH:$SLURM_SUBMIT_DIR/libs/"
echo $PYTHONPATH

echo "Running experiment"
MLFLOW_TRACKING_URI=http://acireale.iiar.pwr.edu.pl/mlflow/ mlflow run . -P trials=10000 -P rmpx-size=11 -P hash=md5 -P agent=yacs -P modulo=8

echo "OK"
