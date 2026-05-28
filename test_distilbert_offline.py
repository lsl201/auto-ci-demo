#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from transformers import DistilBertTokenizer, DistilBertModel

# 离线模型绝对路径
MODEL_PATH = "/home/ubuntu/Desktop/ai-models/distilbert-base-chinese"

# 路径与文件检查
if not os.path.isdir(MODEL_PATH):
    raise SystemExit(f"❌ 路径不存在：{MODEL_PATH}")
for f in ["config.json", "vocab.txt"]:
    if not os.path.exists(os.path.join(MODEL_PATH, f)):
        raise SystemExit(f"❌ 缺少文件：{f}")

print("✅ 路径与文件检查通过")

# 1. 分词器加载（去掉 use_auth_token）
tokenizer = DistilBertTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    cache_dir=None
)

# 2. 模型加载（去掉 use_auth_token）
model = DistilBertModel.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    cache_dir=None,
    ignore_mismatched_sizes=True  # 加上这行，警告就消失了
)

print("✅ distilbert-base-chinese 加载成功（Python3）")

# 测试推理
inputs = tokenizer("测试文本", return_tensors="pt")
outputs = model(**inputs)
print("✅ 推理测试通过，输出形状：", outputs.last_hidden_state.shape)