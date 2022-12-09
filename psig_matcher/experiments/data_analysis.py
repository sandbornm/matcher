import mlflow

mlflow_client = mlflow.client.MlflowClient()
experiment_id = mlflow_client.get_experiment_by_name("Experiment 2").experiment_id
print(experiment_id)
runs = mlflow_client.search_runs(experiment_id)
counter = 0

for r in runs:
    counter += 1
    print("run_id: {}".format(r.info.run_id))
    print("lifecycle_stage: {}".format(r.info.lifecycle_stage))
    print("metrics: {}".format(r.data.metrics))

print(counter)
