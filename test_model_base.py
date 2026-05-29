import pytest
import mlflow
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataQualityPreset, DataDriftPreset

def load_model_sample_data(model_info):
    data = [
        {"input_text": "今天天气不错", "embedding_norm": 0.82},
        {"input_text": "测试句子", "embedding_norm": 0.77},
        {"input_text": "人工智能很好玩", "embedding_norm": 0.91},
    ]
    return pd.DataFrame(data)

def test_model_mlflow_logging(model_info):
    MODEL_NAME = model_info["name"]
    MODEL_PATH = model_info["path"]

    with mlflow.start_run(run_name=f"{MODEL_NAME}_base_test"):
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("model_path", MODEL_PATH)
        df = load_model_sample_data(model_info)

        mlflow.log_metric("sample_count", len(df))
        mlflow.log_metric("embedding_norm_mean", df["embedding_norm"].mean())

def test_model_data_quality_and_drift(model_info):
    df = load_model_sample_data(model_info)

    report = Report(metrics=[
        DataQualityPreset(),
        DataDriftPreset()
    ])
    report.run(reference_data=df, current_data=df)

    assert not df.empty
    assert df["input_text"].isnull().sum() == 0
    assert df["embedding_norm"].isnull().sum() == 0

def test_model_basic_stats(model_info):
    df = load_model_sample_data(model_info)
    assert not df.empty
    assert df["embedding_norm"].min() >= 0