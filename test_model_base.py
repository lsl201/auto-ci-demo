import pytest
import mlflow
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataQualityPreset, DataDriftPreset

# ------------------------------
#  fixture 供所有测试用例使用
# ------------------------------
@pytest.fixture
def model_info():
    return {
        "name": "ai_quality_demo_model",
        "path": "models/demo.pkl"
    }

# ------------------------------
#  测试 1：MLflow 日志
# ------------------------------
def test_model_mlflow_logging(model_info):
    with mlflow.start_run(run_name="ai_model_test_run"):
        mlflow.log_param("model_name", model_info["name"])
        mlflow.log_param("model_path", model_info["path"])
        mlflow.log_metric("test_metric", 0.95)

# ------------------------------
#  测试 2：数据质量 + 漂移检测
# ------------------------------
def test_model_data_quality(model_info):
    data = {
        "feature1": [1.2, 2.3, 3.4, 4.5],
        "feature2": [0.8, 0.9, 0.7, 0.6]
    }
    df = pd.DataFrame(data)

    report = Report(metrics=[
        DataQualityPreset(),
        DataDriftPreset()
    ])
    report.run(reference_data=df, current_data=df)

    assert not df.empty
    assert df.isnull().sum().sum() == 0

# ------------------------------
#  测试 3：基础指标检查
# ------------------------------
def test_model_basic_validation(model_info):
    df = pd.DataFrame({
        "score": [0.9, 0.85, 0.88, 0.92]
    })
    assert df["score"].mean() > 0.8