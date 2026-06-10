# 【避坑提醒（针对你的情况）：】
# 1、不要死磕数学和理论：
# 作为转型者，你的优势是工程化落地能力，不是算法理论。能看懂指标、会用工具、能解释结果，就足够应付大部分岗位了。
# 2、不要贪多求全：
# 每个工具都学一遍不如把3个核心工具（MLflow/Evidently/Hugging Face Evaluate）学透，做出一个完整的项目。
# 3、每学一个工具，都要输出可展示的成果：
# 比如MLflow的实验记录、Evidently的漂移报告、LIME的可解释性分析图，这些都是你简历里的硬通货，比你学了多少理论更有用。

# 数据漂移 + Evidently 报表 + MLflow 归档 + 质量门禁断言
import os
import pytest
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
import allure

# ==========配置区（按需修改路径，对应你的CSV、离线数据集）==========
BASE_PATH = "./data/"
REF_CSV = os.path.join(BASE_PATH, "base_dataset.csv")   # 基准数据集（历史正常样本）
CURR_CSV = os.path.join(BASE_PATH, "online_test.csv")    # 线上待测数据集
DRIFT_HTML = "./report/drift_report.html"
DRIFT_THRESHOLD = 0.3  # 漂移特征占比门禁：超过30%判定漂移不合格

@pytest.fixture(scope="module")
def drift_dataset():
    """加载基准/当前数据集fixture，复用至全用例"""
    df_ref = pd.read_csv(REF_CSV)
    df_curr = pd.read_csv(CURR_CSV)
    # 字段映射：文本特征+标签（适配你的AI分类数据）
    col_map = ColumnMapping()
    col_map.text_features = ["input_text"]
    col_map.target = "label"
    return df_ref, df_curr, col_map

@allure.feature("数据漂移检测｜Evidently")
@allure.story("数据集整体漂移+单特征漂移门禁校验")
def test_dataset_drift_check(drift_dataset):
    df_ref, df_curr, col_map = drift_dataset
    os.makedirs("./report", exist_ok=True)

    # 1、生成Evidently漂移报告
    drift_report = Report(metrics=[DataDriftPreset(drift_share=DRIFT_THRESHOLD)])
    drift_report.run(reference_data=df_ref, current_data=df_curr, column_mapping=col_map)
    drift_report.save_html(DRIFT_HTML)
    res = drift_report.as_dict()["metrics"][0]["result"]

    # 2、MLflow埋点：指标+HTML报告归档
    with mlflow.start_run(run_name="drift_check_run"):
        drift_ratio = res["dataset_drift_share"]
        mlflow.log_metric("drift_feature_ratio", drift_ratio)
        mlflow.log_param("drift_threshold", DRIFT_THRESHOLD)
        mlflow.log_artifact(DRIFT_HTML, artifact_path="drift_html")

    # 3、质量门禁：断言漂移超标直接失败，阻断CI发布
    assert drift_ratio < DRIFT_THRESHOLD, f"漂移特征占比{drift_ratio:.2f}≥阈值{DRIFT_THRESHOLD},数据漂移超标"