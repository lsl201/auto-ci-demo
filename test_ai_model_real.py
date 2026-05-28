import pytest
import os
from transformers import AutoTokenizer, AutoModel
import torch

# 模型路径（和你的目录完全对应）
MODEL_DIR = os.path.expanduser("~/Desktop/ai-models/all-MiniLM-L6-v2")

def test_model_offline_load():
    """测试模型能否完全离线加载（关键！不依赖网络）"""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )
    assert tokenizer is not None
    assert model is not None
    print("✅ 模型&分词器离线加载成功")

def test_sentence_embedding_inference():
    """测试真实推理，生成句子向量"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)

    # 测试两个句子，验证向量生成
    test_sentences = [
        "Jenkins AI质量测试流水线",
        "AI模型自动化测试用例"
    ]

    for sent in test_sentences:
        inputs = tokenizer(sent, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # 取句子向量（维度固定为384）
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        assert embedding.shape[0] == 384
        print(f"✅ 句子「{sent}」推理成功，向量维度：384")

if __name__ == "__main__":
    test_model_offline_load()
    test_sentence_embedding_inference()