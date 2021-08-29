import tempfile
import mlflow
import dill

def log_object(filename, obj):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/{filename}"
        with open(path, mode="w") as f:
            dill.dump(obj, path)
            # f.flush()
        mlflow.log_artifact(path)
