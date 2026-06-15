# test_ood_sample.py
# OOD域外样本鲁棒性测试
import pytest
import mlflow
import allure
import pandas as pd
from tests.common.model_loader import get_offline_model

# ======================== 加载测试数据 ========================
def get_base_samples():
    df = pd.read_csv("/home/ubuntu/Desktop/auto-ci-demo/tests/data/offline_test_dataset.csv")
    df = df[df["input_text"].str.strip() != ""].reset_index(drop=True)
    return df.to_dict("records")

base_samples = get_base_samples()

# ======================== OOD域外样本：按场景分组构造 ========================
def create_ood_samples():
    """
    OOD域外样本分组：全部和手机评论业务无关
    1.日常天气问答
    2.菜谱烹饪问题
    3.汽车养护类内容
    4.游戏资讯内容
    5.商品价格咨询（非手机）
    6.无意义重复汉字
    7.纯随机英文字母乱码
    """
    group_dict = {}

    group_dict["weather_qa"] = [
        "明天天气怎么样"
    ]
    group_dict["cook_recipe"] = [
        "西红柿炒鸡蛋怎么做"
    ]
    group_dict["car_maintain"] = [
        "如何保养汽车轮胎"
    ]
    group_dict["game_info"] = [
        "王者荣耀最新出装"
    ]
    group_dict["price_ask"] = [
        "苹果手机多少钱"
    ]
    group_dict["repeat_nonsense"] = [
        "啊啊啊啊啊啊啊啊啊"
    ]
    group_dict["random_letter"] = [
        "qwertyuiop"
    ]

    return group_dict

# ======================== OOD鲁棒测试用例 ========================
@allure.epic("AI模型鲁棒性测试")
@allure.feature("OOD域外样本测试")
@allure.story("无关文本不能误判为业务正类")
@allure.title("OOD测试：{raw_data[input_text]}")
@pytest.mark.robust_ood
@pytest.mark.risk_legal
@pytest.mark.parametrize("raw_data", base_samples)
def test_ood_robust(raw_data):
    model = get_offline_model()
    text = raw_data["input_text"]
    true_label = raw_data["true_label"]

    # 中英场景映射（Allure中文步骤 + MLflow中文TAG）
    mapping = {
        "weather_qa": "日常天气问答OOD",
        "cook_recipe": "菜谱烹饪问题OOD",
        "car_maintain": "汽车养护内容OOD",
        "game_info": "游戏资讯内容OOD",
        "price_ask": "非手机价格咨询OOD",
        "repeat_nonsense": "无意义重复汉字OOD",
        "random_letter": "纯随机字母乱码OOD"
    }

    with mlflow.start_run(run_name=f"OOD测试_{text[:18]}"):
        mlflow.log_param("robust_test_type", "OOD_sample")
        mlflow.log_param("original_text", text[:30])

        with allure.step("步骤1：按场景分组加载OOD域外测试样本"):
            ood_group_dict = create_ood_samples()

        total_wrong = 0
        total_cnt = 0

        with allure.step("步骤2：各组OOD域外样本批量预测"):
            for ood_type, sample_list in ood_group_dict.items():
                cn_name = mapping[ood_type]
                with allure.step(f"OOD场景：{cn_name}，本组样本数{len(sample_list)}"):
                    pred_list = [model.predict([x])[0] for x in sample_list]
                    # OOD判定规则：域外样本不该被判为正类1，p==1即为误判
                    wrong_count = sum(1 for p in pred_list if p == 1)
                    group_error_rate = wrong_count / len(sample_list)

                    # 单组参数、细分指标落盘
                    mlflow.log_param(f"{ood_type}_sample_count", len(sample_list))
                    mlflow.log_metric(f"{ood_type}_wrong_cnt", wrong_count)
                    mlflow.log_metric(f"{ood_type}_error_rate", group_error_rate)
                    # MLflow追加中文注释TAG
                    mlflow.set_tag(f"{ood_type}_cn_desc", cn_name)

                    total_wrong += wrong_count
                    total_cnt += len(sample_list)

        with allure.step("步骤3：统计全局OOD误判数量与误判率"):
            ood_error_rate = total_wrong / total_cnt

        with allure.step("步骤4：MLflow全局指标落盘"):
            # 【原有全局指标key完全原样保留，不修改】
            mlflow.log_metric("ood_wrong_count", total_wrong)
            mlflow.log_metric("ood_error_rate", ood_error_rate)

        # 原有质量门禁完全不动
        assert ood_error_rate <= 0.2, f"OOD误判超标: {ood_error_rate:.2%}"