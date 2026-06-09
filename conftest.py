import os, shutil
import gc, torch
import pytest

# 全局会话启动只执行一次，所有用例批量运行只清一次allure
def pytest_configure(config):
    dir_allure = "./allure-results"
    if os.path.exists(dir_allure):
        shutil.rmtree(dir_allure)
    os.makedirs(dir_allure, exist_ok=True)


@pytest.fixture(autouse=True)
def free_mem():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===================== 【新增：只执行一次指标汇总】 =====================
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_sessionfinish(session, exitstatus):
    """
    所有用例全部跑完后，只执行 1 次汇总
    不管跑1个、6个、全部用例，都只调用一次 metric_summary.py
    """
    yield

    print("\n" + "="*60)
    print("📊 所有用例执行完毕 → 统一汇总 MLflow 指标")
    print("="*60)

    script_path = os.path.join(os.getcwd(), "scripts/metric_summary.py")
    if os.path.exists(script_path):
        import subprocess
        subprocess.run(
            ["python", script_path],
            check=False,
            cwd=os.getcwd()
        )
    else:
        print(f"⚠️  未找到汇总脚本：{script_path}")