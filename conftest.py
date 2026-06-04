import os,shutil

# 全局会话启动只执行一次，所有用例批量运行只清一次allure
def pytest_configure(config):
    dir_allure = "./allure-results"
    if os.path.exists(dir_allure):
        shutil.rmtree(dir_allure)
    os.makedirs(dir_allure, exist_ok=True)