# mlflow_text_class.py（优化版：增加数据量，提高准确率）
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# ===================== 离线模型路径 =====================
MODEL_PATH = "/home/ubuntu/Desktop/ai-models/all-MiniLM-L6-v2"
# ========================================================

# 1. 扩充后的文本数据（太空 vs 宗教，共40条）
texts = [
    # 太空类 (标签1)
    "The spacecraft launched successfully to Mars orbit.",
    "NASA is planning a new mission to study black holes.",
    "The new telescope captured amazing images of distant galaxies.",
    "Astronauts trained for years before their space mission.",
    "Scientists are looking for signs of life on other planets.",
    "The space station orbits Earth every 90 minutes.",
    "The rocket engine ignited with a powerful roar.",
    "Deep space exploration requires advanced technology.",
    "The moon landing was a historic achievement.",
    "Black holes have extremely strong gravitational pull.",
    "Stars are born in vast clouds of gas and dust.",
    "The James Webb telescope sees the earliest galaxies.",
    "Astronauts conduct experiments in zero gravity.",
    "SpaceX successfully landed its reusable rocket.",
    "Mars rover sends back stunning images of the red planet.",
    # 宗教类 (标签0)
    "Many people believe in God and practice their religion.",
    "The debate about the existence of God has been ongoing.",
    "Different religions have different views on the afterlife.",
    "Religious texts have been translated into many languages.",
    "The church bell rang to mark the start of the service.",
    "Prayer is an important part of many people's daily lives.",
    "Hinduism is one of the world's oldest major religions.",
    "Buddhism teaches the path to enlightenment.",
    "People gather in mosques for Friday prayers.",
    "The Torah is a sacred text in Judaism.",
    "Religious holidays are celebrated with various traditions.",
    "Monks meditate to achieve inner peace.",
    "The Quran is the central religious text of Islam.",
    "People of faith often find comfort in their beliefs.",
    "Many churches feature beautiful stained glass windows."
]
labels = [1]*15 + [0]*15

# 2. 划分训练集/测试集 (80%训练, 20%测试)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. 加载本地离线BERT模型
embedding_model = SentenceTransformer(MODEL_PATH, local_files_only=True)

# 4. 文本转向量
X_train_vec = embedding_model.encode(X_train)
X_test_vec = embedding_model.encode(X_test)

# ===================== MLflow 训练 + 记录 =====================
# ✅ 这一行必须加！连接 Jenkins 里启动的 MLflow 服务
mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("文本分类实战（离线BERT版）")

with mlflow.start_run(run_name="local_bert_train_v2"):
    params = {
        "feature_model": "all-MiniLM-L6-v2 (离线)",
        "classifier": "LogisticRegression",
        "C": 1.0
    }
    mlflow.log_params(params)

    # 训练分类器
    model = LogisticRegression(C=1.0, solver="liblinear")
    model.fit(X_train_vec, y_train)

    # 测试指标
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    # 保存模型并注册
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="text_classifier_model",
        registered_model_name="text-classifier"
    )

    print("="*50)
    print(f"✅ 训练完成！")
    print(f"🎯 准确率：{acc:.4f}")
    print(f"📦 模型已存入 MLflow：text-classifier")
    print("="*50)