https://apps.plgrid.pl/module/plgrid/tools/anaconda


https://kdm.cyfronet.pl/portal/Podstawy:SLURM#Uruchamianie_zada.C5.84_wsadowych


## dev

    conda env create -f environment.yml
    conda activate rmpx-hashing-benchmark
    pip install -r requirements.txt

    MLFLOW_TRACKING_URI=http://acireale.iiar.pwr.edu.pl/mlflow/ mlflow run . -P trials=10000 -P rmpx-size=11 -P hash=md5 -P agent=yacs -P modulo=8

## Zeus cluster

Build package with instructions and push it to broker server
    
    make slurm_tar
    scp slurm-delivery.tar.gz zeus:~

On broker server:

    tar -xzvf slurm-delivery.tar.gz
    sbatch slurm/slurm.sh
    tail -f output.out -f error.err
