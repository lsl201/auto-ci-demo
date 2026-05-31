import pytest
from sklearn.metrics import accuracy_score

@pytest.mark.regression
def test_model_output_dim(model_fixture):
    """回归：校验模型输出向量维度符合预期"""
    tokenizer = model_fixture["tokenizer"]
    model = model_fixture["model"]
    expect_dim = model_fixture["dim"]

    text = "回归测试文本"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    vec = outputs.last_hidden_state[:, 0, :]

    assert vec.shape[-1] == expect_dim

@pytest.mark.regression
def test_full_accuracy(test_data_csv, metrics_threshold):
    """回归：全量样本准确率校验"""
    y_true = test_data_csv["y_true"]
    y_pred = test_data_csv["y_pred"]
    acc = accuracy_score(y_true, y_pred)
    assert acc >= metrics_threshold["accuracy"], f"准确率不达标，当前值：{acc:.2f}"