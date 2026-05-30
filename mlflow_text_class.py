# mlflow_text_class.py（离线版：使用你本地 all-MiniLM-L6-v2 模型）
import mlflow
import mlflow.sklearn
import numpy as np
import os
import shutil
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
# ---------------- 自动清理旧实验：只保留当前实验，删除其他 ----------------
from mlflow.tracking import MlflowClient

# 你要保留的实验名称（和后面 set_experiment 一致）
KEEP_EXP_NAME = "文本分类实战（离线BERT版）"

client = MlflowClient()

# 列出所有非 deleted 的实验
all_exps = mlflow.search_experiments(view_type=mlflow.entities.ViewType.ACTIVE_ONLY)

for exp in all_exps:
    exp_id = exp.experiment_id
    exp_name = exp.name
    # 跳过要保留的实验
    if exp_name == KEEP_EXP_NAME:
        print(f"✅ 保留实验：{exp_name} (ID:{exp_id})")
        continue
    # 删除其他所有实验
    print(f"🗑️  删除旧实验：{exp_name} (ID:{exp_id})")
    client.delete_experiment(exp_id)
# ---------------------------------------------------------------------------

mlruns_dir = "mlruns"
if os.path.exists(mlruns_dir):
    for exp_id in os.listdir(mlruns_dir):
        # 只删数字ID文件夹，跳过你要保留的那个ID
        if exp_id.isdigit() and exp_id != "2":  # 把"2"换成你保留实验的ID
            exp_path = os.path.join(mlruns_dir, exp_id)
            shutil.rmtree(exp_path)
            print(f"🧹 清理磁盘旧目录：{exp_path}")

# ===================== 你的离线模型真实路径 =====================
MODEL_PATH = "/home/ubuntu/Desktop/ai-models/all-MiniLM-L6-v2"
# ==============================================================

# 1. 模拟文本数据
texts = [
    "The spacecraft launched successfully to Mars orbit.",
    "NASA is planning a new mission to study black holes.",
    "The new telescope captured amazing images of distant galaxies.",
    "Many people believe in God and practice their religion.",
    "The debate about the existence of God has been ongoing for centuries.",
    "Different religions have different views on the afterlife.",
    "Astronauts trained for years before their space mission.",
    "Scientists are looking for signs of life on other planets.",
    "Religious texts have been translated into many languages.",
    "The space station orbits Earth every 90 minutes."
]
labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 1]

# 2. 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

# 3. 加载【本地离线模型】（关键！）
embedding_model = SentenceTransformer(MODEL_PATH, local_files_only=True)

# 4. 文本转向量
X_train_vec = embedding_model.encode(X_train)
X_test_vec = embedding_model.encode(X_test)

# 5. MLflow 实验
mlflow.set_experiment("文本分类实战（离线BERT版）")
with mlflow.start_run(run_name="use_local_bert_model"):
    # 记录参数（会显示你用了离线模型）
    params = {
        "feature_model": "all-MiniLM-L6-v2 (本地离线)",
        "local_model_path": MODEL_PATH,
        "classifier": "LogisticRegression",
        "C": 1.0
    }
    mlflow.log_params(params)

    # 训练
    model = LogisticRegression(C=1.0, solver="liblinear")
    model.fit(X_train_vec, y_train)

    # 评估
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    # 保存模型
    mlflow.sklearn.log_model(model, "classifier_model")

    print(f"✅ 运行完成！使用本地离线模型：{MODEL_PATH}")
    print(f"🎯 模型准确率：{acc:.4f}")