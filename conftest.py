import os
import shutil
import gc
import torch
import pytest
import mlflow

# 全局清空allure结果目录
def pytest_configure(config):
    dir_allure = "./allure-results"
    if os.path.exists(dir_allure):
        shutil.rmtree(dir_allure)
    os.makedirs(dir_allure, exist_ok=True)

    # 1️⃣ MLflow全局配置：设置tracking uri，不在这里写死experiment
    mlflow.set_tracking_uri("file:///home/ubuntu/Desktop/auto-ci-demo/mlruns")

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
    # 遍历所有用例，找到 test_99_final_attach_csv
    for item in items:
        if "test_99_final_attach_csv" in item.nodeid:
            last_item = item
            break
    # 找到就挪到最后
    if last_item:
        items.remove(last_item)
        items.append(last_item)

# 2️⃣ 新增：根据用例路径自动选择MLflow Experiment，并设置test_type标签
@pytest.fixture(autouse=True, scope="function")
def set_mlflow_experiment_and_tag(request):
    """
    功能：
    1. 根据用例文件路径，自动切换到不同的MLflow Experiment
    2. 给run自动打test_type标签，和汇总脚本里的字段对应
    """
    file_path = request.node.fspath
    test_type = "unknown"
    exp_name = "model-test-suite"  # 默认兜底实验

    # 判断用例属于哪个类型
    if "metrics" in str(file_path):
        # 离线指标用例（test_offline_model_metric.py）
        exp_name = "model-test-suite-offline"
        test_type = "offline"
    elif "robustness" in str(file_path):
        # 鲁棒性用例（噪声/对抗/漂移/OOD/公平性）
        exp_name = "model-test-suite-robustness"
        test_type = "robustness"

    # 设置MLflow实验
    mlflow.set_experiment(exp_name)

    # 提前绑定tag，后续用例内部start_run自动继承
    # mlflow.set_tag("test_type", test_type)
    yield test_type