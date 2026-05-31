# tests/conftest.py
import sys
import os

# --------------------------
# 关键：把项目根目录加入 Python 路径
# --------------------------
# 当前文件路径：tests/conftest.py
# 向上一级就是项目根目录
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# --------------------------
# 修正导入路径
# 因为 common 是 tests 下的子包，所以必须写 tests.common
# --------------------------
from tests.common.fixtures import (
    model_fixture,
    minilm_model,
    distilbert_model,
    test_data_csv,
    train_data,
    online_data,
    project_config,
    metrics_threshold,
)