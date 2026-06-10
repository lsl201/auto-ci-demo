# test_text_class.py（AI质量测试：从MLflow拉模型测试）
import mlflow
import mlflow.pyfunc
import numpy as np
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
import pytest  # 1. 新增：导入 pytest

# ===================== 配置 =====================
MODEL_PATH = "/home/ubuntu/Desktop/ai-models/all-MiniLM-L6-v2"
MLFLOW_MODEL_NAME = "text-classifier"
# =================================================

# 2. 新增：把所有代码放进 test_ 开头的函数里 + 添加 @pytest.mark.mine
@pytest.mark.mine
def test_ai_model_quality():

    # 测试数据（独立数据，不参与训练）
    test_texts = [
        "New rocket launched to space station",
        "People pray in church on Sunday",
        "Astronauts walk in space",
        "Holy book is read by believers",
        "SpaceX lands rocket successfully"
    ]
    test_labels = [1, 0, 1, 0, 1]

    # 1. 加载离线BERT向量化模型
    embedding_model = SentenceTransformer(MODEL_PATH, local_files_only=True)
    test_vec = embedding_model.encode(test_texts)

    # ✅ 必须加：连接 Jenkins 里启动的 MLflow 服务
    mlflow.set_tracking_uri("http://localhost:5000")

    # 2. 从 MLflow 加载【最新模型】
    model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}/latest")

    # 3. 预测
    y_pred = model.predict(test_vec)
    acc = accuracy_score(test_labels, y_pred)

    # 4. 输出测试报告
    print("\n===== 🧪 AI 质量测试报告 =====")
    print(f"模型：{MLFLOW_MODEL_NAME} (最新版)")
    print(f"测试准确率：{acc:.4f}")
    print(f"预测结果：{y_pred}")
    print(f"真实标签：{test_labels}")

    # 5. 质量门禁
    PASS_THRESHOLD = 0.8
    
    # 3. 修改：把 if / else 换成 pytest 的 assert
    assert acc >= PASS_THRESHOLD, f"❌ 测试失败：模型质量不达标！准确率={acc:.4f}"

    print("✅ 测试通过：模型质量合格")