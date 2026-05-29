import pytest
from transformers import (
    BertTokenizer, BertModel,
    DistilBertTokenizer, DistilBertModel
)

# 模型配置列表，新增模型仅在此处添加配置项
MODEL_LIST = [
    {
        "name": "minilm",
        "path": "/home/ubuntu/Desktop/ai-models/all-MiniLM-L6-v2",
        "tokenizer_cls": BertTokenizer,
        "model_cls": BertModel,
        "dim": 384,
        "ignore_mismatch": False
    },
    {
        "name": "distilbert_zh",
        "path": "/home/ubuntu/Desktop/ai-models/distilbert-base-chinese",
        "tokenizer_cls": BertTokenizer,
        "model_cls": BertModel,
        "dim": 768,
        "ignore_mismatch": False
    }
]

@pytest.fixture(params=MODEL_LIST, ids=[x["name"] for x in MODEL_LIST])
def model_info(request):
    return request.param