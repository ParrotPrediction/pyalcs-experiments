VER=9

build-project-images:
	docker build -t rmpx-benchmark:${VER} rmpx-benchmarking
	docker tag rmpx-benchmark:${VER} rmpx-benchmark:latest

run:
	docker run --network="host" rmpx-benchmark:latest -P rmpx-size=20 -P trials=100 -P bins=3 -P agent=DynaQ --no-conda
