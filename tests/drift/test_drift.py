import pytest
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

@pytest.mark.drift
def test_data_drift(train_data, online_data, metrics_threshold):
    """数据漂移专项测试：校验整体漂移分数"""
    # 初始化漂移报告
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=train_data, current_data=online_data)

    # 提取漂移结果
    result = drift_report.as_dict()
    drift_result = result["metrics"][0]["result"]

    # 兼容不同Evidently版本的字段名
    if "dataset_drift_score" in drift_result:
        drift_score = drift_result["dataset_drift_score"]
    elif "drift_score" in drift_result:
        drift_score = drift_result["drift_score"]
    else:
        # 兜底方案：直接取是否发生漂移的标志
        dataset_drift = drift_result.get("dataset_drift", False)
        assert not dataset_drift, "数据漂移超标：已检测到整体数据分布漂移"
        return

    # 断言：漂移分数低于阈值即为合格
    assert drift_score < metrics_threshold["drift_score"], \
        f"数据漂移超标，当前分数：{drift_score:.2f}"