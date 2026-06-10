# tests/my_custom_tests/test_user_demo.py
import pytest

# 打自定义标签，方便单独执行
@pytest.mark.mycase
def test_use_base_fixture(project_config, metrics_threshold):
    """使用项目配置、指标阈值 fixture"""
    print("项目模型根路径：", project_config["model_root"])
    assert metrics_threshold["accuracy"] == 0.80
    assert metrics_threshold["drift_score"] <= 0.2

@pytest.mark.mycase
def test_use_data_fixture(test_data_csv):
    """使用测试数据 fixture"""
    # 简单校验数据非空
    assert not test_data_csv.empty
    print("测试数据行数：", len(test_data_csv))

@pytest.mark.mycase
def test_use_single_model(minilm_model):
    """使用单模型 fixture（minilm）"""
    tokenizer = minilm_model["tokenizer"]
    model = minilm_model["model"]
    assert tokenizer is not None
    assert model is not None
    print("MiniLM 模型加载正常")

# 也可以复用原有标签，并入冒烟/回归流程
@pytest.mark.smoke
def test_smoke_custom(distilbert_model):
    """自定义冒烟用例，复用原有标签"""
    assert distilbert_model["model"] is not None