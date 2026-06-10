import pytest

@pytest.mark.smoke
def test_minilm_load(minilm_model):
    """冒烟：校验MiniLM模型、分词器正常加载"""
    assert minilm_model["model"] is not None
    assert minilm_model["tokenizer"] is not None

@pytest.mark.smoke
def test_distilbert_load(distilbert_model):
    """冒烟：校验中文DistilBERT模型、分词器正常加载"""
    assert distilbert_model["model"] is not None
    assert distilbert_model["tokenizer"] is not None

@pytest.mark.smoke
def test_model_predict_simple(model_fixture):
    """冒烟：校验模型基础推理无报错"""
    tokenizer = model_fixture["tokenizer"]
    model = model_fixture["model"]
    text = "测试推理"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    # 校验输出不为空
    assert outputs.last_hidden_state is not None