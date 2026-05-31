import pytest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

@pytest.mark.metrics
def test_all_metrics(test_data_csv, metrics_threshold):
    """批量校验全部模型评估指标"""
    y_true = test_data_csv["y_true"]
    y_pred = test_data_csv["y_pred"]
    y_proba = test_data_csv["y_proba"]

    # 计算指标
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    # 断言阈值
    assert acc >= metrics_threshold["accuracy"], f"准确率 {acc:.2f} 不达标"
    assert pre >= metrics_threshold["precision"], f"精确率 {pre:.2f} 不达标"
    assert rec >= metrics_threshold["recall"], f"召回率 {rec:.2f} 不达标"
    assert f1 >= metrics_threshold["f1"], f"F1 {f1:.2f} 不达标"
    assert auc >= metrics_threshold["auc"], f"AUC {auc:.2f} 不达标"