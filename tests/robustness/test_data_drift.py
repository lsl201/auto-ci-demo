# 数据分布偏移（缺失、缩放、异常值）
# test_data_drift.py
# 数据分布偏移鲁棒性测试（缺失值、缩放、异常值、极端长度、空值）
import pytest
import mlflow
import allure
import pandas as pd
from tests.common.model_loader import get_offline_model


# ===================== 加载测试数据 =====================
def get_base_samples():
    df = pd.read_csv("/home/ubuntu/Desktop/auto-ci-demo/tests/data/offline_test_dataset.csv")
    df = df[df["input_text"].str.strip() != ""].reset_index(drop=True)
    return df.to_dict("records")

base_samples = get_base_samples()

# ===================== 数据分布偏移：构造偏移样本 =====================
def create_drift_samples(text):
    """
    按漂移场景分组构造样本
    1.空值类 2.极端长度 3.纯数字符号 4.跨领域陌生样本 5.业务外长尾无关文本
    """
    group_dict = {}

    # 1、空值/空白类
    group_dict["empty_blank"] = [
        "",
        "   "
    ]

    # 2、极端长度偏移
    group_dict["extreme_len"] = [
        text[:1],          # 极短1个字
        text * 10          # 超长重复10倍
    ]

    # 3、纯数字、特殊符号异常值
    group_dict["symbol_num_noise"] = [
        "1234567890",
        "@@@@@@####$$$%%%^^&&&&",
        "手1机2评3论4混5乱6字符7码",
    ]

    # ========== 需求新增：新领域陌生样本、业务外长尾样本 ==========
    # 假设你的业务是手机评价，下面是完全无关领域文本
    group_dict["unfamiliar_domain"] = [
        "今天股市大盘涨跌如何，基金要不要加仓",
        "孩子小升初择校怎么填报志愿",
        "汽车保养换机油需要多少工时费"
    ]
    group_dict["long_tail_outside"] = [
        "天气多云，傍晚适合出门散步买菜",
        "快递单号在哪里查看物流轨迹"
    ]

    return group_dict

# ===================== 数据偏移鲁棒测试用例 =====================
@allure.epic("AI模型鲁棒性测试")
@allure.feature("数据分布偏移测试")
@allure.story("缺失值、缩放、异常值、极端长度鲁棒性")
@allure.title("数据偏移测试：{txt_snip}")
@pytest.mark.robust_drift
@pytest.mark.risk_legal
@pytest.mark.parametrize("raw_data", base_samples)
def test_data_drift_robust(raw_data):
    model = get_offline_model()
    text = raw_data["input_text"]
    true_label = raw_data["true_label"]

    with mlflow.start_run(run_name=f"数据漂移测试_{text[:18]}"):
        mlflow.log_param("robust_test_type", "data_drift")
        mlflow.log_param("original_text", text[:30])

        # ==========【新增：中英场景映射字典，只写一次】==========
        mapping = {
            "empty_blank": "空值/空白文本",
            "extreme_len": "极端长度文本(过短/超长)",
            "symbol_num_noise": "数字+特殊符号混杂脏输入",
            "unfamiliar_domain": "跨领域陌生域外样本",
            "long_tail_outside": "业务外长尾无关样本"
        }
        
        with allure.step("步骤1：按场景分组生成数据偏移样本"):
            drift_group_dict = create_drift_samples(text)

        total_hit = 0
        total_cnt = 0

        with allure.step("步骤2：各组漂移样本批量预测"):
            for drift_type, sample_list in drift_group_dict.items():
                cn_name = mapping[drift_type]
                with allure.step(f"漂移场景：{cn_name}，本组样本数{len(sample_list)}"):
                    pred_list = [model.predict([x])[0] for x in sample_list]
                    hit = sum(1 for p in pred_list if p == true_label)
                    group_rate = hit / len(sample_list)

                    # 每组参数、细分指标单独入库
                    mlflow.log_param(f"{drift_type}_sample_count", len(sample_list))
                    mlflow.log_metric(f"{drift_type}_robust_rate", group_rate)
                    mlflow.log_metric(f"{drift_type}_error_rate", 1 - group_rate)

                    # ==========【循环内插入这一行】写入中文描述TAG ==========
                    mlflow.set_tag(f"{drift_type}_cn_desc", mapping[drift_type])
                    
                    total_hit += hit
                    total_cnt += len(sample_list)

        with allure.step("步骤3：计算全局漂移鲁棒率"):
            drift_robust_rate = total_hit / total_cnt

        with allure.step("步骤4：汇总全局指标记录"):
            mlflow.log_metric("drift_robust_rate", drift_robust_rate)
            mlflow.log_metric("drift_error_rate", 1 - drift_robust_rate)

        # 原有门禁断言完全不动
        assert drift_robust_rate >= 0.6, f"数据偏移鲁棒率不达标: {drift_robust_rate:.2%}"