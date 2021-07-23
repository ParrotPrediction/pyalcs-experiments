https://apps.plgrid.pl/module/plgrid/tools/anaconda


https://kdm.cyfronet.pl/portal/Podstawy:SLURM#Uruchamianie_zada.C5.84_wsadowych


## dev

    conda env create -f conda.yml
    conda activate rmpx-hashing-benchmark
    pip install -r requirements.txt

    MLFLOW_TRACKING_URI=http://acireale.iiar.pwr.edu.pl/mlflow/ mlflow run . -P trials=10000 -P rmpx-size=11 -P hash=md5 -P agent=yacs -P modulo=8

## zeus

load python module
load anaconda module

plgrid/tools/python

pip install mlflow

Build package with instructions and push it to broker server
    
    make slurm_zip
    scp slurm-delivery.zip zeus:~

On broker server:

    unzip slurm-delivery.zip
