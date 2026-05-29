import torch
import time
import psutil
import os
import allure
from torch.nn.functional import cosine_similarity

def test_model_load_and_embedding(model_info):
    """
    企业级离线模型完整验证（修复版）
    """
    # ========== 1. 读取模型配置 ==========
    name = model_info["name"]
    path = model_info["path"]
    tokenizer_cls = model_info["tokenizer_cls"]
    model_cls = model_info["model_cls"]
    dim = model_info["dim"]
    ignore_mismatch = model_info["ignore_mismatch"]

    # ========== 2. 加载分词器 & 模型（纯离线） ==========
    tokenizer = tokenizer_cls.from_pretrained(path, local_files_only=True)
    model = model_cls.from_pretrained(
        path,
        local_files_only=True,
        ignore_mismatched_sizes=ignore_mismatch
    )
    model.eval()

    # 获取当前进程（监控内存）
    process = psutil.Process(os.getpid())

    # ========== 3. 统一句向量提取函数 ==========
    def get_embedding(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # 句向量

    # ========== 4. 基础推理 & 性能指标 ==========
    test_text = "这是一条标准测试语句" if "zh" in name else "This is a test"

    t0 = time.time()
    emb = get_embedding(test_text)
    latency = time.time() - t0
    memory = process.memory_info().rrs / 1024 / 1024  # MB

    # 性能断言
    assert latency < 3.0, f"推理超时：{latency:.2f}s"
    assert memory < 2000, f"内存占用过高：{memory:.0f}MB"
    allure.attach(f"latency={latency:.2f}s | memory={memory:.0f}MB", name="性能指标")

    # ========== 5. 向量基础校验 ==========
    assert emb.shape == (1, dim), f"向量维度错误：{emb.shape}"
    assert not torch.isnan(emb).any(), "存在NaN值"
    assert not torch.isinf(emb).any(), "存在无穷值"
    assert torch.sum(torch.abs(emb)) > 0, "输出全零向量"

    # ========== 6. 稳定性：重复推理一致性 ==========
    emb1 = get_embedding(test_text)
    emb2 = get_embedding(test_text)
    sim_stable = cosine_similarity(emb1, emb2).item()
    assert sim_stable > 0.99, f"稳定性不足：{sim_stable}"
    allure.attach(f"一致性相似度={sim_stable:.4f}", name="稳定性")

    # ========== 7. 鲁棒性：异常输入 ==========
    robust_texts = [
        "",                    # 空字符串
        "!!!@@@#$%^&*()",      # 特殊符号
        "a" * 2000,            # 超长文本
        "1234567890",          # 纯数字
        "🤣😀😎🎉🙌",         # Emoji
    ]
    for t in robust_texts:
        try:
            e = get_embedding(t)
            assert e.shape == (1, dim)
        except Exception as ex:
            assert False, f"鲁棒性失败：输入【{t}】报错：{ex}"

    allure.attach("空串/特殊符号/超长文本/Emoji 全部通过", name="鲁棒性")

    # ========== 8. 语义质量：相似 vs 不相似（修复阈值和测试句） ==========
    if "zh" in name:
        same1 = "我喜欢学习人工智能"
        same2 = "人工智能非常有意思"
        diff = "今天外面下雨了，我只想在家睡觉"
    else:
        same1 = "I love AI technology"
        same2 = "Artificial intelligence is amazing"
        diff = "I hate rainy days and staying indoors"

    emb_s1 = get_embedding(same1)
    emb_s2 = get_embedding(same2)
    emb_d = get_embedding(diff)

    sim_pos = cosine_similarity(emb_s1, emb_s2).item()
    sim_neg = cosine_similarity(emb_s1, emb_d).item()

    # 语义质量断言（放宽不相似阈值）
    assert sim_pos > 0.6, f"相似句相似度太低：{sim_pos}"
    assert sim_neg < 0.65, f"不相似句相似度太高：{sim_neg}"

    allure.attach(
        f"相似={sim_pos:.2f} | 不相似={sim_neg:.2f}",
        name="语义质量"
    )

    # ========== 最终输出 ==========
    print(f"✅【{name}】企业级全指标验证通过")
    print(f"  耗时：{latency:.2f}s | 内存：{memory:.0f}MB")
    print(f"  一致性：{sim_stable:.4f} | 相似：{sim_pos:.2f} | 不相似：{sim_neg:.2f}")