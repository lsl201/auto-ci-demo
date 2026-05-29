import torch

def test_model_load_and_embedding(model_info):
    """统一测试：模型加载 + 推理 + 向量有效性校验"""
    name = model_info["name"]
    path = model_info["path"]
    tokenizer_cls = model_info["tokenizer_cls"]
    model_cls = model_info["model_cls"]
    dim = model_info["dim"]
    ignore_mismatch = model_info["ignore_mismatch"]

    # 加载分词器与模型（纯离线）
    tokenizer = tokenizer_cls.from_pretrained(path, local_files_only=True)
    model = model_cls.from_pretrained(
        path,
        local_files_only=True,
        ignore_mismatched_sizes=ignore_mismatch
    )
    model.eval()

    # 区分中英文测试文本
    test_text = "这是测试语句" if "zh" in name else "This is a test"
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True)

    # 推理计算
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取句向量
    emb = outputs.last_hidden_state[:, 0, :]

    # 多层断言校验模型功能
    assert emb.shape == (1, dim), f"【{name}】向量维度不匹配"
    assert torch.sum(torch.abs(emb)) > 0, f"【{name}】输出向量全为0"
    assert not torch.isnan(emb).any(), f"【{name}】存在无效NaN值"
    assert not torch.isinf(emb).any(), f"【{name}】存在无穷值"

    print(f"✅ {name} 模型校验通过 | 向量维度：{dim}")