# test_noise_input.py
# 鲁棒性测试 - 脏输入扰动测试
import pytest
import mlflow
import allure
import pandas as pd
from tests.common.model_loader import get_offline_model
exp_name = "model-test-suite"
exp_id = mlflow.set_experiment(exp_name)

# ===================== 加载你的离线数据集（完美匹配） =====================
def get_base_samples():
    df = pd.read_csv("/home/ubuntu/Desktop/auto-ci-demo/tests/data/offline_test_dataset.csv")
    df = df[df["input_text"].str.strip() != ""].reset_index(drop=True)
    return df.to_dict("records")

base_samples = get_base_samples()

# ===================== 脏数据构造（手机评论业务适配） =====================
def add_noise(s):
    variants = [
        "",                  
        "   ",               
        "@@##$$",            
        s + "!!!",       
        s[:1],               
        s.replace(" ", ""),  
        "😂😂" + s,          
    ]
    return variants

# ===================== 测试用例 =====================
@allure.epic("AI模型鲁棒性测试")
@allure.feature("脏输入扰动测试")
@allure.story("噪声、空值、乱码鲁棒测试")
@allure.title("测试脏输入鲁棒性：{raw_data[input_text]}")
@pytest.mark.robust_noise
@pytest.mark.risk_legal
@pytest.mark.parametrize("raw_data", base_samples)
def test_noise_robust(raw_data):
    model = get_offline_model()

    # 完美匹配你的字段：input_text, true_label
    text = raw_data["input_text"]
    true_label = raw_data["true_label"]

    with mlflow.start_run(run_name=f"脏输入测试_{text[:18]}"):
        mlflow.log_param("robust_test_type", "noise_input")
        mlflow.log_param("original_text", text[:30])
        mlflow.set_tag("test_type", "robustness")  # 加上这行
        
        with allure.step("步骤1：生成脏数据"):
            noise_list = add_noise(text)

        with allure.step("步骤2：模型预测"):
            pred_list = [model.predict([x])[0] for x in noise_list]

        with allure.step("步骤3：计算鲁棒准确率"):
            hit = sum(1 for p in pred_list if p == true_label)
            robust_accuracy = hit / len(noise_list)

        with allure.step("步骤4：记录MLflow指标"):
            mlflow.log_metric("robust_noise_accuracy", robust_accuracy)
            mlflow.log_metric("robust_noise_error_rate", 1 - robust_accuracy)

        # 质量门禁
    assert robust_accuracy >= 0.6, f"鲁棒性不达标：{robust_accuracy:.2%}"