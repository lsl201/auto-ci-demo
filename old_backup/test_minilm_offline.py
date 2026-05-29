import os
import torch
from transformers import BertTokenizer, BertModel

MODEL_PATH = "/home/ubuntu/Desktop/ai-models/all-MiniLM-L6-v2"

def test_minilm_model_load_and_work():
    assert os.path.isdir(MODEL_PATH), "模型目录不存在"

    # 加载
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = BertModel.from_pretrained(MODEL_PATH, local_files_only=True)

    # 推理
    inputs = tokenizer("This is a test", return_tensors="pt")
    outputs = model(**inputs)

    # --------------------------
    # ✅ 关键：验证功能是否正常
    # --------------------------
    embedding = outputs.last_hidden_state[:, 0, :]  # 句向量

    # 1. 形状必须正确
    assert embedding.shape == torch.Size([1, 384])

    # 2. 不能全0
    assert torch.abs(embedding).sum().item() > 0

    # 3. 数值必须正常（不是无效值）
    assert not torch.isnan(embedding).any()
    assert not torch.isinf(embedding).any()

    print("✅ all-MiniLM 模型加载成功 + 功能完全正常！")