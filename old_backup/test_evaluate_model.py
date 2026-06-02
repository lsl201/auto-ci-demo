import allure
import pytest
import math
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 忽略警告
import warnings
warnings.filterwarnings("ignore")

# ---------------------- 1. 文本分类指标评测 ----------------------
@pytest.mark.model_eval
@allure.feature("大模型评测")
@allure.story("文本分类指标")
@allure.title("测试准确率、F1分数计算")
def test_text_classification_metrics():
    references = [1, 0, 1, 0, 1]
    predictions = [1, 0, 0, 0, 1]

    acc = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average="macro")

    # ========== 修复：必须加 attachment_type=allure.attachment_type.TEXT ==========
    allure.attach(f"准确率: {acc:.2f}", name="分类准确率", attachment_type=allure.attachment_type.TEXT)
    allure.attach(f"F1分数: {f1:.2f}", name="分类F1分数", attachment_type=allure.attachment_type.TEXT)

    print("文本分类指标：")
    print(f"准确率: {acc:.2f}")
    print(f"F1分数: {f1:.2f}")

    assert acc >= 0.5
    assert f1 >= 0.4

# ---------------------- 2. 文本生成评测（BLEU/ROUGE） ----------------------
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
    bleu_final = sum(bleu_scores) / len(bleu_scores)

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge_scores = []
    for ref, pred in zip(references_text, predictions_text):
        score = scorer.score(ref[0], pred)
        rouge_scores.append(score['rouge1'].fmeasure)
    rouge_final = sum(rouge_scores) / len(rouge_scores)

    # ========== 修复：必须加 attachment_type=allure.attachment_type.TEXT ==========
    allure.attach(f"BLEU分数: {bleu_final:.2f}", name="BLEU", attachment_type=allure.attachment_type.TEXT)
    allure.attach(f"ROUGE-1分数: {rouge_final:.2f}", name="ROUGE-1", attachment_type=allure.attachment_type.TEXT)

    print("\n文本生成指标：")
    print(f"BLEU分数: {bleu_final:.2f}")
    print(f"ROUGE-1分数: {rouge_final:.2f}")

    assert bleu_final >= 0.0
    assert rouge_final >= 0.3