import pytest
import mlflow
import pandas as pd
from evidently import Report
from evidently.metrics import ColumnDistributionMetric

# --------------------------
# 数据加载（由 fixture 传入 model_info）
# --------------------------
def load_model_sample_data(model_info):
    """模拟模型离线推理数据：input + output"""
    data = [
        {"input_text": "今天天气不错", "embedding_norm": 0.82},
        {"input_text": "测试句子", "embedding_norm": 0.77},
        {"input_text": "人工智能很好玩", "embedding_norm": 0.91},
    ]
    return pd.DataFrame(data)

# --------------------------
# 通用测试用例（都接收 model_info）
# --------------------------
def test_model_mlflow_logging(model_info):
    """通用：用 MLflow 记录模型基本信息、参数、指标"""
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


def test_model_data_quality(model_info):
    """通用：Evidently 数据质量检查（兼容当前版本）"""
    MODEL_NAME = model_info["name"]
    df = load_model_sample_data(model_info)

    # 必须传入 metrics，这里用最简单的 ColumnDistributionMetric
    report = Report(metrics=[
        ColumnDistributionMetric(column_name="embedding_norm")
    ])
    report.run(current_data=df)

    html_path = f"/tmp/{MODEL_NAME}_data_quality.html"
    report.save_html(html_path)

    # 空值断言（和原来一样）
    assert df["input_text"].isnull().sum() == 0, "input_text 存在空值"
    assert df["embedding_norm"].isnull().sum() == 0, "embedding_norm 存在空值"


def test_model_basic_stats(model_info):
    """通用：基础统计/分布检查（兼容当前版本）"""
    df = load_model_sample_data(model_info)

    # 必须传入 metrics，这里复用 ColumnDistributionMetric
    report = Report(metrics=[
        ColumnDistributionMetric(column_name="embedding_norm")
    ])
    report.run(current_data=df)

    assert df.isnull().sum().sum() == 0, "数据中存在空值"