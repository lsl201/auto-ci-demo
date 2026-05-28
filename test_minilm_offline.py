#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from transformers import BertTokenizer, BertModel

MODEL_PATH = "/home/ubuntu/Desktop/ai-models/all-MiniLM-L6-v2"

def test_minilm_model_load():
    # 检查路径
    assert os.path.isdir(MODEL_PATH), "模型目录不存在"
    
    # 加载
    tokenizer = BertTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, cache_dir=None
    )
    model = BertModel.from_pretrained(
        MODEL_PATH, local_files_only=True, cache_dir=None
    )
    
    # 测试推理
    inputs = tokenizer("test offline model", return_tensors="pt")
    outputs = model(**inputs)
    
    # 断言成功
    assert outputs.last_hidden_state is not None
    print("✅ all-MiniLM-L6-v2 测试通过")