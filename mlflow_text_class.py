# mlflow_text_class.py（免下载版）
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. 模拟文本分类数据（不用联网下载）
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
labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 1]  # 1=space, 0=atheism

# 2. 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

# 3. 文本向量化
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. MLflow 实验配置
mlflow.set_experiment("文本分类实战（作业版）")
with mlflow.start_run(run_name="logreg_baseline_demo"):
    # 记录超参数
    params = {
        "model": "LogisticRegression",
        "max_features": 5000,
        "C": 1.0,
        "solver": "liblinear"
    }
    mlflow.log_params(params)

    # 训练模型
    model = LogisticRegression(C=params["C"], solver=params["solver"])
    model.fit(X_train_tfidf, y_train)

    # 评估+记录指标
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_0", report["0"]["precision"])
    mlflow.log_metric("recall_0", report["0"]["recall"])
    mlflow.log_metric("f1_0", report["0"]["f1-score"])

    # 保存模型（关键：截图要看到 artifacts）
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ 完成！准确率：{acc:.4f}")
    print("👉 查看UI：mlflow ui ，浏览器访问 http://localhost:5000")