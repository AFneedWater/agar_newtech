from __future__ import annotations
from typing import Dict, Any
import mlflow


class MLflowLogger:
    def __init__(self, tracking_uri: str, experiment: str, run_name: str = ""):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.run = mlflow.start_run(run_name=run_name or None)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)

    def close(self) -> None:
        mlflow.end_run()
