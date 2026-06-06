# train_online_sim_model.py 训练脚本，使用你现成tests/data/offline_test_dataset.csv
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1、加载你的测试数据集（和离线模型用同一份数据）
df = pd.read_csv("tests/data/offline_test_dataset.csv")
texts = df["input_text"].tolist()
labels = df["true_label"].tolist()

# 2、搭建轻量化流水线：TF-IDF文本向量化+逻辑回归二分类
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000)), # 控制特征量，模型极小
    ("lr", LogisticRegression(random_state=42))
])

# 3、训练充当「线上部署版本」的小模型
pipe.fit(texts, labels)

# 4、持久化保存模型（后续API服务加载）
joblib.dump(pipe, "./online_sim_model.joblib")
print("模拟线上的小模型训练完成，保存：online_sim_model.joblib")