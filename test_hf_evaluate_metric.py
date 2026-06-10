# 【避坑提醒（针对你的情况）：】
# 1、不要死磕数学和理论：
# 作为转型者，你的优势是工程化落地能力，不是算法理论。能看懂指标、会用工具、能解释结果，就足够应付大部分岗位了。
# 2、不要贪多求全：
# 每个工具都学一遍不如把3个核心工具（MLflow/Evidently/Hugging Face Evaluate）学透，做出一个完整的项目。
# 3、每学一个工具，都要输出可展示的成果：
# 比如MLflow的实验记录、Evidently的漂移报告、LIME的可解释性分析图，这些都是你简历里的硬通货，比你学了多少理论更有用。

# HF-Evaluate 替换手写准确率 / F1，统一评测口径
import pytest
import evaluate
import mlflow
import allure

# 初始化官方评测指标（替换你原有自定义acc计算）
clf_metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
QUALITY_ACC_THRESH = 0.85  # 准确率门禁
QUALITY_F1_THRESH = 0.82   # F1门禁

@pytest.fixture(scope="module")
def predict_label_data():
    """fixture：接口批量预测结果+真实标签（从csv用例读取）"""
    # 实际从csv批量跑接口获取preds, refs
    predictions = [1,0,1,1,0,1,0,1]
    references = [1,0,0,1,0,1,1,1]
    return predictions, references

@allure.feature("模型指标评测｜HF-Evaluate")
@allure.story("标准化Acc/F1/P/R计算+质量门禁")
def test_model_standard_metric(predict_label_data):
    preds, refs = predict_label_data
    # 官方标准指标计算
    metric_res = clf_metric.compute(predictions=preds, references=refs, average="weighted")

    # MLflow全量埋点
    with mlflow.start_run(run_name="hf_evaluate_run"):
        for k,v in metric_res.items():
            mlflow.log_metric(k, round(v,4))

    # 双门禁校验，不达标阻断上线
    assert metric_res["accuracy"] >= QUALITY_ACC_THRESH, f"准确率{metric_res['accuracy']:.3f}<{QUALITY_ACC_THRESH}"
    assert metric_res["f1"] >= QUALITY_F1_THRESH, f"F1{metric_res['f1']:.3f}<{QUALITY_F1_THRESH}"