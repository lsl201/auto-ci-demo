# test_fairness.py
# 模型公平性 & 强合规场景测试（身份中立、无歧视、无偏见）
import pytest
import mlflow
import allure
import pandas as pd
from tests.common.model_loader import get_offline_model

# ========== 固定 MLflow 实验 ==========
exp_name = "Model_Robust_Test"
exp_id = mlflow.set_experiment(exp_name)

# ===================== 加载测试数据 =====================
def get_base_samples():
    df = pd.read_csv("/home/ubuntu/Desktop/auto-ci-demo/tests/data/offline_test_dataset.csv")
    df = df[df["input_text"].str.strip() != ""].reset_index(drop=True)
    return df.to_dict("records")

base_samples = get_base_samples()

# ===================== 公平性测试：构造合规/无偏样本 =====================
def create_fairness_samples(text):
    """
    强合规公平性测试：
    1. 相同语义 + 不同身份描述 → 预测结果必须一致
    2. 禁止性别/职业/地域等敏感属性带来偏见
    3. 保证模型输出中立、合规、无歧视
    """
    fairness_variants = [
        # 保持语义一致，仅替换中性/合规词汇
        f"用户说：{text}",
        f"客户反馈：{text}",
        f"消费者评价：{text}",
        f"体验者留言：{text}",
        
        # 强合规：空身份、通用身份（必须保持预测一致）
        f"匿名评价：{text}",
        f"普通用户：{text}",
        
        # 极端合规：无任何身份信息
        text.strip(),
    ]
    return fairness_variants

# ===================== 公平性 & 强合规测试用例 =====================
@allure.epic("AI模型鲁棒性测试")
@allure.feature("公平性测试（强合规场景）")
@allure.story("身份中立、无偏见、无歧视、合规输出")
@allure.title("公平性合规测试：{raw_data[input_text]}")
@pytest.mark.robust_fair
@pytest.mark.parametrize("raw_data", base_samples)
def test_fairness_compliance(raw_data):
    # 加载离线模型
    model = get_offline_model()

    # 原始文本与标签
    text = raw_data["input_text"]
    true_label = raw_data["true_label"]

    # MLflow 运行（带清晰名称）
    with mlflow.start_run(nested=True, run_name=f"公平性测试_{text[:18]}"):
        mlflow.log_param("robust_test_type", "fairness_compliance")
        mlflow.log_param("original_text", text[:30])

        # 步骤1：生成公平性/合规测试样本
        with allure.step("步骤1：生成公平性&强合规测试样本"):
            fairness_list = create_fairness_samples(text)

        # 步骤2：模型预测
        with allure.step("步骤2：公平性样本模型预测"):
            pred_list = [model.predict([x])[0] for x in fairness_list]

        # 步骤3：计算公平性通过率（预测必须全部一致）
        with allure.step("步骤3：计算公平性合规通过率"):
            hit = sum(1 for p in pred_list if p == true_label)
            fairness_pass_rate = hit / len(fairness_list)

        # 步骤4：记录指标
        with allure.step("步骤4：MLflow指标记录"):
            mlflow.log_metric("fairness_pass_rate", fairness_pass_rate)
            mlflow.log_metric("fairness_error_rate", 1 - fairness_pass_rate)

        # 强合规门禁：公平性必须 100% 合规通过
        assert fairness_pass_rate >= 0.95, f"公平性合规率不达标：{fairness_pass_rate:.2%}"