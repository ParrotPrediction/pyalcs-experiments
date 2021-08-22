image_name = rmpx-benchmark
version=9

build_image:
	docker build -t $(image_name):latest -t $(image_name):${VER} .

interactive:
	docker run --entrypoint bash -ti $(image_name):latest

run_mlflow:
	MLFLOW_TRACKING_URI=http://localhost/mlflow/ mlflow run . $(PARAMS)

run_docker:
	docker run -d --env-file .env --network="host" $(image_name):latest --no-conda $(PARAMS)

slurm_tar:
	tar -czvf slurm-delivery.tar.gz --exclude=*cache* code/* slurm/environment.yml slurm/MLproject slurm/*.sh
