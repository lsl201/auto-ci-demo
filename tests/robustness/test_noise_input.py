# test_noise_input.py
# 鲁棒性测试 - 脏输入扰动测试
import pytest
import mlflow
import allure
import pandas as pd
from tests.common.model_loader import get_offline_model

# ======================== 加载你的离线数据集 ========================
def get_base_samples():
    df = pd.read_csv("/home/ubuntu/Desktop/auto-ci-demo/tests/data/offline_test_dataset.csv")
    df = df[df["input_text"].str.strip() != ""].reset_index(drop=True)
    return df.to_dict("records")

base_samples = get_base_samples()

# ======================== 脏输入噪声扰动：分组构造样本 ========================
def add_noise(text):
    """
    噪声扰动场景分组：
    1.空值空白扰动
    2.特殊符号干扰
    3.尾部追加乱码符号
    4.截断极短文本
    5.删除全部空格
    6.emoji表情插入干扰
    7.随机删字扰动
    8.随机插入冗余字符
    9.语序局部打乱
    """
    group_dict = {}

    # 1、空值空白
    group_dict["empty_blank"] = [
        "",
        "   "
    ]
    # 2、纯特殊符号
    group_dict["symbol_only"] = [
        "@@###$$"
    ]
    # 3、原文尾部追加符号
    group_dict["append_symbol"] = [
        text + "!!!!"
    ]
    # 4、文本截断取首字符
    group_dict["text_cut_short"] = [
        text[:1]
    ]
    # 5、剔除全部空格
    group_dict["remove_space"] = [
        text.replace(" ", "")
    ]
    # 6、插入emoji表情干扰
    group_dict["emoji_noise"] = [
        "😡😒" + text
    ]
    # 7、随机删字扰动（简单实现：删掉第2个字）
    if len(text) >= 2:
        del_char_text = text[:1] + text[2:]
    else:
        del_char_text = text
    group_dict["random_del_char"] = [del_char_text]
    # 8、随机插字扰动
    group_dict["random_insert_char"] = [
        text[:2] + "冗余字" + text[2:]
    ]
    # 9、简单语序打乱（分句倒置）
    if "，" in text:
        parts = text.split("，")
        shuffle_text = parts[-1] + "，" + "，".join(parts[:-1])
    else:
        shuffle_text = text
    group_dict["word_order_shuffle"] = [shuffle_text]

    return group_dict

# ======================== 测试用例 ========================
@allure.epic("AI模型鲁棒性测试")
@allure.feature("脏输入扰动测试")
@allure.story("噪声、空值、乱码鲁棒测试")
@allure.title("测试脏输入鲁棒性：{raw_data[input_text]}")
@pytest.mark.robust_noise
@pytest.mark.risk_legal
@pytest.mark.parametrize("raw_data", base_samples)
def test_noise_robust(raw_data):
    model = get_offline_model()

    text = raw_data["input_text"]
    true_label = raw_data["true_label"]

    # 中英场景映射，用于Allure中文步骤+MLflow中文TAG
    mapping = {
        "empty_blank": "空值空白扰动",
        "symbol_only": "纯特殊符号干扰",
        "append_symbol": "尾部追加符号乱码",
        "text_cut_short": "文本截断极短输入",
        "remove_space": "剔除全部空格",
        "emoji_noise": "Emoji表情插入干扰",
        "random_del_char": "随机删字扰动",
        "random_insert_char": "随机插入冗余字符",
        "word_order_shuffle": "局部语序打乱扰动"
    }

    with mlflow.start_run(run_name=f"脏输入测试_{text[:18]}"):
        mlflow.log_param("robust_test_type", "noise_input")
        mlflow.log_param("original_text", text[:30])

        with allure.step("步骤1：按场景分组生成噪声脏输入样本"):
            noise_group_dict = add_noise(text)

        total_hit = 0
        total_cnt = 0

        with allure.step("步骤2：各组噪声样本批量预测"):
            for noise_type, sample_list in noise_group_dict.items():
                cn_name = mapping[noise_type]
                with allure.step(f"噪声场景：{cn_name}，本组样本数{len(sample_list)}"):
                    pred_list = [model.predict([x])[0] for x in sample_list]
                    hit = sum(1 for p in pred_list if p == true_label)
                    group_acc = hit / len(sample_list)

                    # 每组独立参数+细分指标落盘
                    mlflow.log_param(f"{noise_type}_sample_count", len(sample_list))
                    mlflow.log_metric(f"{noise_type}_robust_acc", group_acc)
                    mlflow.log_metric(f"{noise_type}_error_rate", 1 - group_acc)
                    # MLflow写入中文注释TAG
                    mlflow.set_tag(f"{noise_type}_cn_desc", cn_name)

                    total_hit += hit
                    total_cnt += len(sample_list)

        with allure.step("步骤3：计算全局噪声鲁棒准确率"):
            robust_accuracy = total_hit / total_cnt

        with allure.step("步骤4：记录MLflow全局指标"):
            # 原有全局指标key完全不变，兼容旧汇总脚本
            mlflow.log_metric("robust_noise_accuracy", robust_accuracy)
            mlflow.log_metric("robust_noise_error_rate", 1 - robust_accuracy)

        # 原有质量门禁完全保留，无修改
        assert robust_accuracy >= 0.6, f"鲁棒性不达标: {robust_accuracy:.2%}"