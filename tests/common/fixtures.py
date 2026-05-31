import os
import pytest
import pandas as pd
from transformers import (
    BertTokenizer, BertModel,
    DistilBertTokenizer, DistilBertModel
)

# ==========================
# 项目全局配置
# ==========================
@pytest.fixture(scope="session")
def project_config():
    return {
        "model_root": "/home/ubuntu/Desktop/ai-models",
        "test_data_path": "tests/data/test_cases.csv",
        "train_data_path": "tests/data/train_data.csv",
        "online_data_path": "tests/data/online_data.csv",
    }

# ==========================
# 模型指标阈值（测试通过标准）
# ==========================
@pytest.fixture(scope="session")
def metrics_threshold():
    return {
        "accuracy": 0.80,
        "precision": 0.65,
        "recall": 0.85,
        "f1": 0.80,
        "auc": 0.88,
        "drift_score": 0.2,
    }

# ==========================
# 测试数据 Fixture
# ==========================
@pytest.fixture(scope="session")
def test_data_csv(project_config):
    return pd.read_csv(project_config["test_data_path"])

@pytest.fixture(scope="session")
def train_data(project_config):
    return pd.read_csv(project_config["train_data_path"])

@pytest.fixture(scope="session")
def online_data(project_config):
    return pd.read_csv(project_config["online_data_path"])

# ==========================
# 模型配置
# ==========================
MODEL_LIST = [
    {
        "name": "minilm",
        "folder": "all-MiniLM-L6-v2",
        "tokenizer_cls": BertTokenizer,
        "model_cls": BertModel,
        "dim": 384,
        "ignore_mismatch": False
    },
    {
        "name": "distilbert_zh",
        "folder": "distilbert-base-chinese",
        "tokenizer_cls": DistilBertTokenizer,
        "model_cls": DistilBertModel,
        "dim": 768,
        "ignore_mismatch": False
    }
]

# ==========================
# 多模型并行测试 Fixture
# ==========================
@pytest.fixture(params=MODEL_LIST, ids=[x["name"] for x in MODEL_LIST], scope="session")
def model_fixture(request, project_config):
    cfg = request.param
    model_path = os.path.join(project_config["model_root"], cfg["folder"])

    if not os.path.isdir(model_path):
        raise NotADirectoryError(f"模型不存在：{model_path}")

    tokenizer = cfg["tokenizer_cls"].from_pretrained(
        model_path, local_files_only=True, ignore_mismatched_sizes=cfg["ignore_mismatch"]
    )
    model = cfg["model_cls"].from_pretrained(
        model_path, local_files_only=True, ignore_mismatched_sizes=cfg["ignore_mismatch"]
    )

    return {
        "name": cfg["name"],
        "path": model_path,
        "tokenizer": tokenizer,
        "model": model,
        "dim": cfg["dim"]
    }

# ==========================
# 单模型 Fixture（冒烟测试）
# ==========================
@pytest.fixture(scope="session")
def minilm_model(project_config):
    path = os.path.join(project_config["model_root"], "all-MiniLM-L6-v2")
    tok = BertTokenizer.from_pretrained(path, local_files_only=True)
    model = BertModel.from_pretrained(path, local_files_only=True)
    return {"tokenizer": tok, "model": model}

@pytest.fixture(scope="session")
def distilbert_model(project_config):
    path = os.path.join(project_config["model_root"], "distilbert-base-chinese")
    tok = DistilBertTokenizer.from_pretrained(path, local_files_only=True)
    model = DistilBertModel.from_pretrained(path, local_files_only=True)
    return {"tokenizer": tok, "model": model}