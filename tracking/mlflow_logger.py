import mlflow


def setup_mlflow():

    mlflow.set_experiment(
        "SafeRL-Driving"
    )

    mlflow.start_run()
