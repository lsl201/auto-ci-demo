import os
import shutil
import gc
import torch
import pytest

# 全局清空allure结果目录
def pytest_configure(config):
    dir_allure = "./allure-results"
    if os.path.exists(dir_allure):
        shutil.rmtree(dir_allure)
    os.makedirs(dir_allure, exist_ok=True)

# 自动释放显存
@pytest.fixture(autouse=True)
def free_mem():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 关键：跨目录也能把 test_99_final_attach_csv 固定到最后
def pytest_collection_modifyitems(session, config, items):
    last_item = None
    # 遍历所有用例，找到 test_99
    for item in items:
        if "test_99_final_attach_csv" in item.nodeid:
            last_item = item
            break
    # 找到就挪到最后
    if last_item:
        items.remove(last_item)
        items.append(last_item)