#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from transformers import DistilBertTokenizer, DistilBertModel

MODEL_PATH = "/home/ubuntu/Desktop/ai-models/distilbert-base-chinese"

def test_distilbert_model_load():
    # 检查路径
    assert os.path.isdir(MODEL_PATH), "模型目录不存在"
    
    # 加载
    tokenizer = DistilBertTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, cache_dir=None
    )
    model = DistilBertModel.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        cache_dir=None,
        ignore_mismatched_sizes=True
    )
    
    # 测试推理
    inputs = tokenizer("测试离线模型", return_tensors="pt")
    outputs = model(**inputs)
    
    # 断言成功
    assert outputs.last_hidden_state is not None
    print("✅ distilbert-base-chinese 测试通过")