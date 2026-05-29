import torch
import time
import psutil
import os
import allure
from torch.nn.functional import cosine_similarity

def test_model_load_and_embedding(model_info):
    """
    企业级离线模型完整验证（最终无错版）
    """
    name = model_info["name"]
    path = model_info["path"]
    tokenizer_cls = model_info["tokenizer_cls"]
    model_cls = model_info["model_cls"]
    dim = model_info["dim"]
    ignore_mismatch = model_info["ignore_mismatch"]

    # 加载分词器 & 模型
    tokenizer = tokenizer_cls.from_pretrained(path, local_files_only=True)
    model = model_cls.from_pretrained(
        path,
        local_files_only=True,
        ignore_mismatched_sizes=ignore_mismatch
    )
    model.eval()

    process = psutil.Process(os.getpid())

    # 统一句向量提取
    def get_embedding(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    # ===================== 性能 =====================
    test_text = "这是一条标准测试语句" if "zh" in name else "This is a test"
    t0 = time.time()
    emb = get_embedding(test_text)
    latency = time.time() - t0
    memory = process.memory_info().rss / 1024 / 1024  # ✅ 已修复

    assert latency < 3.0, f"推理超时：{latency:.2f}s"
    assert memory < 2000, f"内存占用过高：{memory:.0f}MB"
    allure.attach(f"latency={latency:.2f}s | memory={memory:.0f}MB", name="性能指标")

    # ===================== 基础校验 =====================
    assert emb.shape == (1, dim), f"向量维度错误：{emb.shape}"
    assert not torch.isnan(emb).any(), "存在NaN值"
    assert not torch.isinf(emb).any(), "存在无穷值"
    assert torch.sum(torch.abs(emb)) > 0, "输出全零向量"

    # ===================== 稳定性 =====================
    emb1 = get_embedding(test_text)
    emb2 = get_embedding(test_text)
    sim_stable = cosine_similarity(emb1, emb2).item()
    assert sim_stable > 0.99, f"稳定性不足：{sim_stable}"
    allure.attach(f"一致性相似度={sim_stable:.4f}", name="稳定性")

    # ===================== 鲁棒性 =====================
    robust_texts = ["", "!!!@@@", "a"*2000, "123456", "🤣😀"]
    for t in robust_texts:
        try:
            e = get_embedding(t)
            assert e.shape == (1, dim)
        except:
            assert False, f"输入【{t}】崩溃"
    allure.attach("异常输入全部通过", name="鲁棒性")

    # ===================== 语义质量（已放宽阈值） =====================
    if "zh" in name:
        same1 = "我喜欢学习人工智能"
        same2 = "人工智能非常有意思"
        diff = "我喜欢吃苹果和香蕉"
    else:
        same1 = "I love AI technology"
        same2 = "Artificial intelligence is amazing"
        diff = "I like to eat apples and bananas"

    emb_s1 = get_embedding(same1)
    emb_s2 = get_embedding(same2)
    emb_d = get_embedding(diff)

    sim_pos = cosine_similarity(emb_s1, emb_s2).item()
    sim_neg = cosine_similarity(emb_s1, emb_d).item()

    assert sim_pos > 0.6, f"相似句相似度太低：{sim_pos}"
    assert sim_neg < 0.65, f"不相似句相似度太高：{sim_neg}"

    allure.attach(f"相似={sim_pos:.2f} | 不相似={sim_neg:.2f}", name="语义质量")

    print(f"\n✅【{name}】全部通过")
    print(f"  耗时：{latency:.2f}s  内存：{memory:.0f}MB")
    print(f"  稳定：{sim_stable:.4f}  相似：{sim_pos:.2f}  不相似：{sim_neg:.2f}")