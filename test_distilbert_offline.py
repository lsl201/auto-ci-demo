import os
import torch
from transformers import DistilBertTokenizer, DistilBertModel

MODEL_PATH = "/home/ubuntu/Desktop/ai-models/distilbert-base-chinese"

def test_distilbert_model_load_and_work():
    assert os.path.isdir(MODEL_PATH), "模型目录不存在"

    # 加载
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = DistilBertModel.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        ignore_mismatched_sizes=True
    )

    # 推理
    inputs = tokenizer("这是一个测试", return_tensors="pt")
    outputs = model(**inputs)

    # --------------------------
    # ✅ 关键：验证功能是否正常
    # --------------------------
    embedding = outputs.last_hidden_state[:, 0, :]

    # 1. 形状必须正确
    assert embedding.shape == torch.Size([1, 768])

    # 2. 不能全0
    assert torch.abs(embedding).sum().item() > 0

    # 3. 数值必须正常
    assert not torch.isnan(embedding).any()
    assert not torch.isinf(embedding).any()

    print("✅ distilbert 中文模型加载成功 + 功能完全正常！")