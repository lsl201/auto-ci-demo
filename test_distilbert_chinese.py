import pytest
import os
from transformers import AutoTokenizer, AutoModel

MODEL_DIR = os.path.expanduser("~/Desktop/ai-models/bert-base-chinese")

def test_bert_offline_load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)
    assert tokenizer is not None
    assert model is not None
    print("✅ bert-base-chinese 离线加载成功")

def test_chinese_inference():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)
    test_text = "人工智能测试流水线"
    inputs = tokenizer(test_text, return_tensors="pt")
    outputs = model(**inputs)
    print(f"✅ 句子推理成功，向量维度：{outputs.last_hidden_state.shape}")

if __name__ == "__main__":
    test_bert_offline_load()
    test_chinese_inference()