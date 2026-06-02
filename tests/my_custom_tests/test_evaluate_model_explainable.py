import allure
import pytest
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 可解释性
import shap
from lime.lime_text import LimeTextExplainer

# 生成指标
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings("ignore")

# 临时图片目录
IMG_DIR = "./tmp_img"
os.makedirs(IMG_DIR, exist_ok=True)

@pytest.mark.model_eval
@allure.feature("大模型评测")
@allure.story("文本分类指标 + 模型可解释性")
@allure.title("测试准确率、F1、SHAP特征重要性、LIME单样本解释")
def test_text_classification_metrics():
    texts = [
        "The movie is good",
        "The movie is bad",
        "Nice film",
        "Terrible acting",
        "Great story"
    ]
    references = [1, 0, 1, 0, 1]

    # 训练流水线
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(random_state=42))
    ])
    model.fit(texts, references)
    predictions = model.predict(texts)

    acc = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average="macro")

    # ========== SHAP 全局特征重要性 + 绘图 ==========
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    X_tf = tfidf.transform(texts)
    feature_names = tfidf.get_feature_names_out()

    explainer_shap = shap.LinearExplainer(clf, X_tf, feature_perturbation="interventional")
    shap_values = explainer_shap.shap_values(X_tf)
    shap_import = np.abs(shap_values).mean(0)

    # 取top10绘图
    top_idx = np.argsort(shap_import)[::-1][:10]
    top_feat = [feature_names[i] for i in top_idx]
    top_val = [shap_import[i] for i in top_idx]

    plt.figure(figsize=(10,4))
    plt.barh(top_feat, top_val)
    plt.gca().invert_yaxis()
    plt.title("SHAP Top10 Feature Importance")
    shap_img_path = os.path.join(IMG_DIR, "shap_feature.png")
    plt.savefig(shap_img_path, bbox_inches="tight")
    plt.close()

    # SHAP文本报告
    shap_report = "SHAP 特征重要性 Top10：\n"
    for f,v in zip(top_feat,top_val):
        shap_report += f"{f}: {v:.4f}\n"

    # 附件：图片+文本
    allure.attach.file(shap_img_path, name="SHAP特征重要性图表", attachment_type=allure.attachment_type.PNG)
    allure.attach(shap_report, name="SHAP特征文本说明", attachment_type=allure.attachment_type.TEXT)

    # ========== LIME 局部解释 + 保存图片 ==========
    explainer_lime = LimeTextExplainer(class_names=["负面","正面"])
    exp = explainer_lime.explain_instance(texts[0], model.predict_proba, num_features=5)
    lime_img_path = os.path.join(IMG_DIR, "lime_sample.png")
    exp.save_to_file(lime_img_path.replace(".png",".html")) # 原始html
    fig = exp.as_pyplot_figure()
    plt.savefig(lime_img_path, bbox_inches="tight")
    plt.close()

    lime_report = f"LIME解释样本：{texts[0]}\n"
    for feat,w in exp.as_list():
        lime_report += f"{feat}: {w:.4f}\n"

    allure.attach.file(lime_img_path, name="LIME单样本解释图", attachment_type=allure.attachment_type.PNG)
    allure.attach(lime_report, name="LIME文本说明", attachment_type=allure.attachment_type.TEXT)

    # 基础指标附件
    allure.attach(f"准确率: {acc:.2f}", name="分类准确率", attachment_type=allure.attachment_type.TEXT)
    allure.attach(f"F1分数: {f1:.2f}", name="分类F1分数", attachment_type=allure.attachment_type.TEXT)

    print(f"acc:{acc:.2f}, f1:{f1:.2f}")
    assert acc >= 0.5
    assert f1 >= 0.4

@pytest.mark.model_eval
@allure.feature("大模型评测")
@allure.story("文本生成指标")
@allure.title("测试BLEU、ROUGE分数计算")
def test_text_generation_metrics():
    predictions_text = ["The movie is good", "The acting is bad"]
    references_text = [["The movie is great"], ["The acting is terrible"]]

    smooth = SmoothingFunction().method4
    bleu_scores = []
    for ref, pred in zip(references_text, predictions_text):
        ref_tokens = ref[0].split()
        pred_tokens = pred.split()
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
        bleu_scores.append(bleu)
    bleu_final = sum(bleu_scores)/len(bleu_scores)

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge_scores = []
    for ref,pred in zip(references_text,predictions_text):
        score = scorer.score(ref[0],pred)
        rouge_scores.append(score["rouge1"].fmeasure)
    rouge_final = sum(rouge_scores)/len(rouge_scores)

    allure.attach(f"BLEU:{bleu_final:.2f}", name="BLEU", attachment_type=allure.attachment_type.TEXT)
    allure.attach(f"ROUGE-1:{rouge_final:.2f}", name="ROUGE-1", attachment_type=allure.attachment_type.TEXT)

    print(f"BLEU:{bleu_final:.2f}, ROUGE:{rouge_final:.2f}")
    assert bleu_final >=0.0
    assert rouge_final >=0.3