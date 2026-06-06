# 数据分布偏移（缺失、缩放、异常值）
# test_data_drift.py
# 数据分布偏移鲁棒性测试（缺失值、缩放、异常值、极端长度、空值）
import pytest
import mlflow
import allure
import pandas as pd
from tests.common.model_loader import get_offline_model

# ========== 固定 MLflow 实验，防止报错 ==========
exp_name = "Model_Robust_Test"
exp_id = mlflow.set_experiment(exp_name)

# ===================== 加载测试数据 =====================
def get_base_samples():
    df = pd.read_csv("/home/ubuntu/Desktop/auto-ci-demo/tests/data/offline_test_dataset.csv")
    df = df[df["input_text"].str.strip() != ""].reset_index(drop=True)
    return df.to_dict("records")

base_samples = get_base_samples()

# ===================== 数据分布偏移：构造偏移样本 =====================
def create_drift_samples(text):
    """
    构造各类数据偏移场景：
    1. 空文本
    2. 纯空格
    3. 极短文本（1个字）
    4. 极长文本（放大10倍）
    5. 全数字
    6. 全符号
    7. 中英文混合乱码
    """
    variants = [
        "",                                  # 空值
        "   ",                               # 纯空格
        text[:1],                            # 极短文本
        text * 10,                           # 极长文本（分布偏移）
        "1234567890",                        # 全数字
        "@@@@@@####$$$%%%^^^&&&",            # 全符号（异常值）
        "手1机2评3论4测5试6乱7码",           # 中英文数字混合异常
    ]
    return variants

# ===================== 数据偏移鲁棒测试用例 =====================
@allure.epic("AI模型鲁棒性测试")
@allure.feature("数据分布偏移测试")
@allure.story("缺失值、缩放、异常值、极端长度鲁棒性")
@allure.title("数据偏移测试：{raw_data[input_text]}")
@pytest.mark.robust_drift
@pytest.mark.risk_legal
@pytest.mark.parametrize("raw_data", base_samples)
def test_data_drift_robust(raw_data):
    # 加载离线模型
    model = get_offline_model()

    # 原始文本与标签
    text = raw_data["input_text"]
    true_label = raw_data["true_label"]

    # MLflow 记录
    with mlflow.start_run(nested=True,run_name=f"数据漂移测试_{text[:18]}"):
        mlflow.log_param("robust_test_type", "data_drift")
        mlflow.log_param("original_text", text[:30])

        # 步骤1：生成分布偏移样本
        with allure.step("步骤1：生成数据偏移样本"):
            drift_list = create_drift_samples(text)

        # 步骤2：模型预测
        with allure.step("步骤2：偏移样本模型预测"):
            pred_list = [model.predict([x])[0] for x in drift_list]

        # 步骤3：计算预测稳定性（分布偏移鲁棒率）
        with allure.step("步骤3：计算鲁棒准确率"):
            hit = sum(1 for p in pred_list if p == true_label)
            drift_robust_rate = hit / len(drift_list)

        # 步骤4：记录指标
        with allure.step("步骤4：MLflow指标记录"):
            mlflow.log_metric("drift_robust_rate", drift_robust_rate)
            mlflow.log_metric("drift_error_rate", 1 - drift_robust_rate)

        # 质量门禁：数据偏移后，预测稳定性 >= 60%
        assert drift_robust_rate >= 0.6, f"数据偏移鲁棒率不达标：{drift_robust_rate:.2%}"