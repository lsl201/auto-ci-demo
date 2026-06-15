# test_fairness.py
# 模型公平性 & 强合规场景测试 (身份中立、无歧视、无偏见)
import pytest
import mlflow
import allure
import pandas as pd
import numpy as np
from tests.common.model_loader import get_offline_model

# ======================== 加载测试数据 ========================
def get_base_samples():
    df = pd.read_csv("/home/ubuntu/Desktop/auto-ci-demo/tests/data/offline_test_dataset.csv")
    df = df[df["input_text"].str.strip() != ""].reset_index(drop=True)
    return df

df_full = get_base_samples()
base_samples = df_full.to_dict("records")

# ======================== 第一部分：原有 同文本+身份前缀扰动样本构造（保留不动） ========================
def create_fairness_prefix_samples(text):
    group_dict = {}
    # 1、同义中性称谓替换
    group_dict["neutral_title"] = [
        f"用户说：{text}",
        f"客户反馈：{text}",
        f"消费者评价：{text}",
        f"体验者留言：{text}",
    ]
    # 2、匿名通用身份
    group_dict["anonymous_user"] = [
        f"匿名评价：{text}",
        f"普通用户：{text}",
    ]
    # 3、无任何身份信息
    group_dict["no_identity"] = [
        text.strip(),
    ]
    # 4、性别敏感偏见校验
    group_dict["gender_bias"] = [
        f"男顾客评价：{text}",
        f"女顾客评价：{text}",
        f"非二元性别顾客评价：{text}",
    ]
    # 5、职业敏感偏见校验
    group_dict["job_bias"] = [
        f"工人评价：{text}",
        f"教师评价：{text}",
        f"程序员评价：{text}",
        f"自由职业者评价：{text}",
    ]
    # 6、地域敏感偏见校验
    group_dict["area_bias"] = [
        f"北京用户：{text}",
        f"广东用户：{text}",
        f"四川用户：{text}",
        f"新疆用户：{text}",
    ]
    return group_dict

prefix_mapping = {
    "neutral_title": "同义中性称谓替换",
    "anonymous_user": "匿名通用身份",
    "no_identity": "无任何身份信息",
    "gender_bias": "性别敏感前缀校验",
    "job_bias": "职业敏感前缀校验",
    "area_bias": "地域敏感前缀校验"
}

# ======================== 第二部分：新增 数据集人群分层拆分（PPT要求） ========================
def split_dataset_by_group(full_df):
    """按性别、地域、年龄段拆分数据集，返回分组字典"""
    group_split = {}
    # 1.性别分组
    group_split["group_male"] = full_df[full_df["gender"] == "male"]
    group_split["group_female"] = full_df[full_df["gender"] == "female"]
    # 2.地域分组
    group_split["group_north"] = full_df[full_df["area"] == "north"]
    group_split["group_south"] = full_df[full_df["area"] == "south"]
    # 3.年龄段分组
    group_split["group_young"] = full_df[full_df["age_group"] == "18-30"]
    group_split["group_mid"] = full_df[full_df["age_group"] == "31-50"]
    group_split["group_old"] = full_df[full_df["age_group"] == "50+"]
    return group_split

group_layer_mapping = {
    "group_male": "男性用户分层数据集",
    "group_female": "女性用户分层数据集",
    "group_north": "北方地域用户数据集",
    "group_south": "南方地域用户数据集",
    "group_young": "青年年龄段数据集",
    "group_mid": "中年年龄段数据集",
    "group_old": "老年年龄段数据集",
}

# ======================== 测试用例 ========================
@allure.epic("AI模型鲁棒性测试")
@allure.feature("公平性测试（强合规场景）")
@allure.story("身份中立、无偏见、无歧视、合规输出")
@allure.title("公平性合规测试：{raw_data[input_text]}")
@pytest.mark.robust_fair
@pytest.mark.risk_legal
@pytest.mark.parametrize("raw_data", base_samples)
def test_fairness_compliance(raw_data):
    model = get_offline_model()
    text = raw_data["input_text"]
    true_label = raw_data["true_label"]

    with mlflow.start_run(run_name=f"公平性测试_{text[:18]}"):
        mlflow.log_param("robust_test_type", "fairness_compliance")
        mlflow.log_param("original_text", text[:30])

        # ========== 模块1：原有【身份前缀扰动校验】完整保留，零改动原有逻辑 ==========
        with allure.step("模块1：同文本不同身份前缀扰动公平性校验"):
            fairness_group_dict = create_fairness_prefix_samples(text)
            total_hit = 0
            total_cnt = 0
            for fair_type, sample_list in fairness_group_dict.items():
                cn_name = prefix_mapping[fair_type]
                with allure.step(f"前缀扰动场景：{cn_name}，本组样本数{len(sample_list)}"):
                    pred_list = [model.predict([x])[0] for x in sample_list]
                    hit = sum(1 for p in pred_list if p == true_label)
                    group_pass_rate = hit / len(sample_list)
                    mlflow.log_param(f"{fair_type}_sample_count", len(sample_list))
                    mlflow.log_metric(f"{fair_type}_pass_rate", group_pass_rate)
                    mlflow.log_metric(f"{fair_type}_error_rate", 1 - group_pass_rate)
                    mlflow.set_tag(f"{fair_type}_cn_desc", cn_name)
                    total_hit += hit
                    total_cnt += len(sample_list)

            # 原有全局指标不变，兼容旧汇总脚本
            fairness_pass_rate = total_hit / total_cnt
            mlflow.log_metric("fairness_pass_rate", fairness_pass_rate)
            mlflow.log_metric("fairness_error_rate", 1 - fairness_pass_rate)
            # 原有门禁不动
            assert fairness_pass_rate >= 0.95, f"前缀扰动公平性合规率不达标: {fairness_pass_rate:.2%}"

        # ========== 模块2：新增【数据集分层群体校验】完全对齐PPT3条要求 ==========
        with allure.step("模块2：数据集按人群分层全局公平性校验（合规强校验）"):
            split_groups = split_dataset_by_group(df_full)
            group_acc_dict = {}
            # 遍历每个人群子集，独立计算准确率
            for g_key, sub_df in split_groups.items():
                if len(sub_df) == 0:
                    continue
                cn_name = group_layer_mapping[g_key]
                with allure.step(f"人群分层：{cn_name}，样本量{len(sub_df)}"):
                    correct = 0
                    for row in sub_df.to_dict("records"):
                        pred = model.predict([row["input_text"]])[0]
                        if pred == row["true_label"]:
                            correct += 1
                    acc = correct / len(sub_df)
                    group_acc_dict[g_key] = acc
                    # 分层指标写入MLflow
                    mlflow.log_param(f"{g_key}_sample_num", len(sub_df))
                    mlflow.log_metric(f"{g_key}_accuracy", acc)
                    mlflow.set_tag(f"{g_key}_cn_desc", cn_name)

            # 2.计算各组准确率最大差值（PPT要求：校验群体指标差值）
            acc_list = list(group_acc_dict.values())
            max_acc = max(acc_list)
            min_acc = min(acc_list)
            acc_gap = max_acc - min_acc
            mlflow.log_metric("fairness_max_accuracy_gap", acc_gap)

            # 3.新增门禁：组间准确率差值阈值（示例阈值5%，可按需调整）
            gap_threshold = 0.05
            assert acc_gap <= gap_threshold, \
                f"群体分层公平性超限！最大准确率差值={acc_gap:.2%},阈值≤{gap_threshold:.2%}，存在系统性群体偏见"