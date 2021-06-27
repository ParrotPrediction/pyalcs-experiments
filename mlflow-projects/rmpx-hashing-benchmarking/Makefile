image_name = rmpx-benchmark

docker_build_image:
	docker build -t $(image_name):latest -t $(image_name):1 .

docker_interactive:
	docker run --entrypoint bash -ti $(image_name):latest

docker_run_experiments:
	docker run -d --rm --env-file .env --network host $(image_name):latest --no-conda $(PARAMS)
