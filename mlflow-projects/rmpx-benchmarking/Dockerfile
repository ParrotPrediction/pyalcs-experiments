FROM python:3.7-slim-buster

ENV MLFLOW_TRACKING_URI="http://localhost/mlflow"

RUN apt-get update \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install mlflow
WORKDIR /experiment

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY MLproject .
COPY code/ code/

ENTRYPOINT ["mlflow", "run", "."]
CMD ["--help"]
