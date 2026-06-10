# 【避坑提醒（针对你的情况）：】
# 1、不要死磕数学和理论：
# 作为转型者，你的优势是工程化落地能力，不是算法理论。能看懂指标、会用工具、能解释结果，就足够应付大部分岗位了。
# 2、不要贪多求全：
# 每个工具都学一遍不如把3个核心工具（MLflow/Evidently/Hugging Face Evaluate）学透，做出一个完整的项目。
# 3、每学一个工具，都要输出可展示的成果：
# 比如MLflow的实验记录、Evidently的漂移报告、LIME的可解释性分析图，这些都是你简历里的硬通货，比你学了多少理论更有用。

# LIME 可解释 + 特征权重图 + MLflow 存图
import os
import pytest
import lime
import lime.lime_text
import matplotlib.pyplot as plt
import mlflow
import allure
import json

# ==========配置（对接你的离线模型/分类接口预测函数）==========
SAVE_IMG = "./report/lime_explain.png"
TEST_SAMPLE = "这款产品性价比很高，使用体验优秀"  # 待解释文本样本

# 【替换成你的模型/接口预测函数：入参文本列表，返回各类别概率】
def model_predict(text_list: list):
    """对接AI分类API/离线模型推理"""
    # 示例：调用你的ai_app_headers请求接口，此处简写
    import requests
    headers = pytest.lazy_fixture("ai_app_headers")
    url = "http://xxx/classify"
    res_list = []
    for txt in text_list:
        resp = requests.post(url, json={"text": txt}, headers=headers)
        res_list.append(resp.json()["prob"])
    return res_list

@pytest.fixture(scope="module")
def lime_explainer():
    explainer = lime.lime_text.LimeTextExplainer(class_names=["负向","正向"])
    return explainer

@allure.feature("模型可解释性｜LIME")
@allure.story("单样本预测关键词权重解析+可视化归档")
def test_model_local_explain(lime_explainer):
    os.makedirs("./report", exist_ok=True)
    explainer = lime_explainer

    # 生成解释结果
    exp = explainer.explain_instance(
        TEST_SAMPLE, model_predict, num_features=8, top_labels=1
    )
    # 保存权重图片
    fig = exp.as_pyplot_figure(label=0)
    plt.tight_layout()
    fig.savefig(SAVE_IMG, dpi=150)
    plt.close()

    # MLflow保存图片+特征权重参数
    with mlflow.start_run(run_name="lime_explain_run"):
        weight_dict = dict(exp.as_list(label=0))
        mlflow.log_params({"sample_text": TEST_SAMPLE, "top_word_weight": json.dumps(weight_dict)})
        mlflow.log_artifact(SAVE_IMG, artifact_path="lime_img")

    # 门禁：关键正向词权重>0判定逻辑合理（自定义业务规则）
    max_weight = max(weight_dict.values())
    assert max_weight > 0, "样本无有效特征权重，模型决策异常"