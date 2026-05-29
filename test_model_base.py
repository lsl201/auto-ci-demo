import pytest
import mlflow
import pandas as pd
from evidently import Report
from evidently.metrics import DatasetDriftMetric
import evidently
print("本地evidently版本:", evidently.__version__)

# --------------------------
# 数据加载
# --------------------------
def load_model_sample_data(model_info):
    data = [
        {"input_text": "今天天气不错", "embedding_norm": 0.82},
        {"input_text": "测试句子", "embedding_norm": 0.77},
        {"input_text": "人工智能很好玩", "embedding_norm": 0.91},
    ]
    return pd.DataFrame(data)

# --------------------------
# 测试用例
# --------------------------
def test_model_mlflow_logging(model_info):
    MODEL_NAME = model_info["name"]
    MODEL_PATH = model_info["path"]

    with mlflow.start_run(run_name=f"{MODEL_NAME}_base_test"):
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("model_path", MODEL_PATH)
        df = load_model_sample_data(model_info)

        mlflow.log_metric("sample_count", len(df))
        mlflow.log_metric("embedding_norm_mean", df["embedding_norm"].mean())

        csv_path = f"/tmp/{MODEL_NAME}_sample_data.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

def test_model_data_quality_and_drift(model_info):
    MODEL_NAME = model_info["name"]
    df = load_model_sample_data(model_info)

    report = Report(metrics=[DatasetDriftMetric()])
    report.run(current_data=df)
    
    html_path = f"/tmp/{MODEL_NAME}_drift.html"
    report.save_html(html_path)
    mlflow.log_artifact(html_path)

    assert df["input_text"].isnull().sum() == 0, "input_text 存在空值"
    assert df["embedding_norm"].isnull().sum() == 0, "embedding_norm 存在空值"

def test_model_basic_stats(model_info):
    df = load_model_sample_data(model_info)
    assert df.isnull().sum().sum() == 0, "数据中存在空值"
    assert df["embedding_norm"].min() >= 0, "embedding_norm 存在负值"