# test_ood_sample.py
# OOD 域外样本鲁棒性测试
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

# ===================== 域外样本（与手机评论无关） =====================
def create_ood_samples():
    return [
        "明天天气怎么样",
        "西红柿炒鸡蛋怎么做",
        "如何保养汽车轮胎",
        "王者荣耀最新出装",
        "苹果手机多少钱",
        "啊啊啊啊啊啊啊啊",
        "qwertyuiop",
    ]

# ===================== OOD鲁棒测试用例 =====================
@allure.epic("AI模型鲁棒性测试")
@allure.feature("OOD域外样本测试")
@allure.story("无关文本不能误判为业务正类")
@allure.title("OOD测试：{raw_data[input_text]}")
@pytest.mark.robust_ood
@pytest.mark.parametrize("raw_data", base_samples)
def test_ood_robust(raw_data):
    # 加载离线模型
    model = get_offline_model()
    text = raw_data["input_text"]
    
    # MLflow 记录
    with mlflow.start_run(nested=True,run_name=f"OOD测试_{text[:18]}"):
        mlflow.log_param("robust_test_type", "OOD_sample")
        
        with allure.step("步骤1：加载域外测试样本"):
            # 1. 生成域外样本
            ood_list = create_ood_samples()

        with allure.step("步骤2：域外样本模型预测"):
            # 2. 模型预测
            pred_list = [model.predict([x])[0] for x in ood_list]

        with allure.step("步骤3：统计误判数量与误判率"):
            # 3. 计算误判率
            wrong_count = sum(1 for p in pred_list if p == 1)
            ood_error_rate = wrong_count / len(ood_list)

        with allure.step("步骤4：MLflow指标落盘"):
            # 4. 记录指标
            mlflow.log_metric("ood_wrong_count", wrong_count)
            mlflow.log_metric("ood_error_rate", ood_error_rate)

        # 5. 质量门禁
        assert ood_error_rate <= 0.2, f"OOD 误判超标：{ood_error_rate:.2%}"