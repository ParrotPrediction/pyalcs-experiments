import mlflow
from time import sleep
import random
import os

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://localhost/mlflow")
    with mlflow.start_run(run_name="a"):
        mlflow.log_param("x", 1)

        if not os.path.exists("output"):
            os.makedirs("output")

        for i in range(100):
            val = random.random()
            mlflow.log_metric("v", val)
            sleep(1)

            if i % 10 == 0:
                with open("output/out.txt", "w") as f:
                    f.write(f"Current val: {val}")

                mlflow.log_artifacts("output")
