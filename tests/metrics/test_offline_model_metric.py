import allure
import pytest
import pandas as pd
import mlflow
import requests
import time
import subprocess
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from tests.common.model_loader import get_offline_model

# ========== 常量 ==========
ONLINE_API_URL = "http://127.0.0.1:8000/predict"

# ========== 计时器 ==========
class Timer:
    def __init__(self):
        self.last = time.time()
    def tick(self, step_name):
        now = time.time()
        cost = round(now - self.last, 2)
        print(f"⏱ {step_name} | 耗时: {cost}s")
        self.last = now

# 封装：批量计算全套指标，复用代码
def calc_all_metrics(y_true, y_pred, suffix=""):
    """
    suffix: 指标后缀，区分全局/normal/edge
    return: 指标字典
    """
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    err_rate = 1 - acc

    res = {
        f"acc{suffix}": acc,
        f"pre{suffix}": pre,
        f"rec{suffix}": rec,
        f"fpr{suffix}": fpr,
        f"fnr{suffix}": fnr,
        f"total_error_rate{suffix}": err_rate
    }
    return res

# ========== 测试用例 ==========
@pytest.mark.metric
@pytest.mark.biz_metric
@allure.feature("离线模型业务指标验收+线上部署衰减对比")
@allure.title("离线vs模拟线上API：准确率&误判率衰减核算")
def test_offline_acc_err_rate(test_type):
    t = Timer()

    with mlflow.start_run(
        run_name="离线vs模拟线上_指标衰减验收",
        tags={"test_type": test_type}
    ):
        # 1. 读取完整数据集
        df = pd.read_csv("tests/data/offline_test_v1.csv")
        mlflow.log_param("total_sample_num", len(df))
        t.tick("读取全部数据")

        # ====================== 新增：子集切分逻辑 ======================
        # 规则：自定义边界样本判定规则（可按需微调）
        def is_edge_sample(text):
            # 判定条件满足任意一条 → 边界极端样本
            if len(text) <= 3:
                return True       # 极短单字/符号
            if any(ch in "！？@#￥%……&*👍🤬🎉🔥📱📸" for ch in text):
                return True       # emoji、特殊符号
            if len(text) >= 150:
                return True       # 超长冗余文本
            # 中性模糊话术关键词
            neutral_keywords = ["一般般", "凑合用", "不好不坏", "中规中矩", "普普通通", "没什么亮点"]
            for kw in neutral_keywords:
                if kw in text:
                    return True
            return False

        # 拆分成两个子集
        df["is_edge"] = df["input_text"].apply(is_edge_sample)
        df_normal = df[df["is_edge"] == False].copy()    # 正常常规样本
        df_edge = df[df["is_edge"] == True].copy()      # 极端边界样本

        mlflow.log_param("normal_sample_num", len(df_normal))
        mlflow.log_param("edge_sample_num", len(df_edge))
        t.tick("数据集拆分为正常/边界子集")

        # 全局全集
        x_full = df["input_text"].tolist()
        y_true_full = df["true_label"].tolist()
        # 正常子集
        x_normal = df_normal["input_text"].tolist()
        y_true_normal = df_normal["true_label"].tolist()
        # 边界子集
        x_edge = df_edge["input_text"].tolist()
        y_true_edge = df_edge["true_label"].tolist()
        # =================================================================

        # 离线模型推理（全集一次性推理，后续子集切片取用预测结果）
        model = get_offline_model()
        t.tick("加载模型")
        y_pred_full = model.predict(x_full)
        t.tick("全局模型推理")

        # 拆分预测结果，对齐两个子集真实标签
        df["pred_label"] = y_pred_full
        y_pred_normal = df[df["is_edge"] == False]["pred_label"].tolist()
        y_pred_edge = df[df["is_edge"] == True]["pred_label"].tolist()

        # ---------------------- 三套指标批量计算 ----------------------
        metrics_full = calc_all_metrics(y_true_full, y_pred_full, suffix="")          # 全局
        metrics_normal = calc_all_metrics(y_true_normal, y_pred_normal, suffix="_normal") # 正常样本
        metrics_edge = calc_all_metrics(y_true_edge, y_pred_edge, suffix="_edge")         # 边界极端样本
        t.tick("全局+正常子集+边界子集指标计算")

        # 输出全局误判样本（原有逻辑不变）
        bad_cases = df[df["true_label"] != df["pred_label"]]
        if len(bad_cases) > 0:
            bad_cases.to_csv("tests/data/bad_case_pool.csv", index=False, encoding="utf-8")
            print(f"✅ 全局误判样本: {len(bad_cases)} 条")
        t.tick("输出误判样本CSV")

        # ---------------------- 线上API推理（仍用全集） ----------------------
        try:
            resp = requests.post(ONLINE_API_URL, json={"text_list": x_full}, timeout=(3,12))
            resp.raise_for_status()
            y_pred_online_full = resp.json()["pred_label"]
        except Exception as e:
            assert False, f"API失败: {e}"
        t.tick("API请求")

        # 线上全集指标
        online_metrics_full = calc_all_metrics(y_true_full, y_pred_online_full, suffix="_on")
        acc_on = online_metrics_full["acc_on"]
        err_on = online_metrics_full["total_error_rate_on"]

        # 衰减计算（沿用原有逻辑）
        decay_acc = acc_on - metrics_full["acc"]
        decay_err = err_on - metrics_full["total_error_rate"]
        decay_fpr = online_metrics_full["fpr_on"] - metrics_full["fpr"]
        decay_fnr = online_metrics_full["fnr_on"] - metrics_full["fnr"]
        decay_pre = online_metrics_full["pre_on"] - metrics_full["pre"]
        decay_rec = online_metrics_full["rec_on"] - metrics_full["rec"]
        t.tick("线上指标&衰减计算")

        # ---------------------- Allure附件（扩充子集指标） ----------------------
        all_data = {
            "off_准确率": f"{metrics_full['acc']:.4f}",
            "off_总误判": f"{metrics_full['total_error_rate']:.4f}",
            "off_FPR": f"{metrics_full['fpr']:.4f}",
            "off_FNR": f"{metrics_full['fnr']:.4f}",
            "off_精确率": f"{metrics_full['pre']:.4f}",
            "off_召回": f"{metrics_full['rec']:.4f}",

            "normal子集_准确率": f"{metrics_normal['acc_normal']:.4f}",
            "normal子集_误判率": f"{metrics_normal['total_error_rate_normal']:.4f}",
            "edge边界子集_准确率": f"{metrics_edge['acc_edge']:.4f}",
            "edge边界子集_误判率": f"{metrics_edge['total_error_rate_edge']:.4f}",

            "on_准确率": f"{acc_on:.4f}",
            "on_总误判": f"{err_on:.4f}",
            "on_FPR": f"{online_metrics_full['fpr_on']:.4f}",
            "on_FNR": f"{online_metrics_full['fnr_on']:.4f}",
            "on_精确率": f"{online_metrics_full['pre_on']:.4f}",
            "on_召回": f"{online_metrics_full['rec_on']:.4f}",

            "准确率衰减": f"{decay_acc:.4f}",
            "误判涨幅": f"{decay_err:.4f}"
        }
        for k, v in all_data.items():
            allure.attach(v, name=k, attachment_type=allure.attachment_type.TEXT)

        # ====================== MLflow批量上报全部指标 ======================
        # 1）全局原有核心指标（和之前完全一致，兼容旧汇总脚本）
        mlflow.log_metric("offline_accuracy", metrics_full["acc"])
        mlflow.log_metric("total_error_rate", metrics_full["total_error_rate"])
        mlflow.log_metric("false_positive_rate_FPR", metrics_full["fpr"])
        mlflow.log_metric("false_negative_rate_FNR", metrics_full["fnr"])
        mlflow.log_metric("precision_off", metrics_full["pre"])
        mlflow.log_metric("recall_off", metrics_full["rec"])

        # 2）正常常规样本子集新增指标
        mlflow.log_metric("offline_accuracy_normal", metrics_normal["acc_normal"])
        mlflow.log_metric("total_error_rate_normal", metrics_normal["total_error_rate_normal"])
        mlflow.log_metric("fpr_normal", metrics_normal["fpr_normal"])
        mlflow.log_metric("fnr_normal", metrics_normal["fnr_normal"])

        # 3）极端边界样本子集新增指标
        mlflow.log_metric("offline_accuracy_edge", metrics_edge["acc_edge"])
        mlflow.log_metric("total_error_rate_edge", metrics_edge["total_error_rate_edge"])
        mlflow.log_metric("fpr_edge", metrics_edge["fpr_edge"])
        mlflow.log_metric("fnr_edge", metrics_edge["fnr_edge"])

        # 4）线上、衰减指标（原有不变）
        mlflow.log_metric("online_accuracy", acc_on)
        mlflow.log_metric("online_total_error", err_on)
        mlflow.log_metric("online_FPR", online_metrics_full["fpr_on"])
        mlflow.log_metric("online_FNR", online_metrics_full["fnr_on"])
        mlflow.log_metric("precision_on", online_metrics_full["pre_on"])
        mlflow.log_metric("recall_on", online_metrics_full["rec_on"])

        mlflow.log_metric("decay_accuracy", decay_acc)
        mlflow.log_metric("decay_err_rate", decay_err)
        mlflow.log_metric("decay_FPR", decay_fpr)
        mlflow.log_metric("decay_FNR", decay_fnr)
        t.tick("全套指标写入MLflow")

        # 原有断言校验不变
        assert metrics_full["acc"] >= 0.90, f"离线全局准确率 {metrics_full['acc']:.2%} <90%"
        assert decay_acc >= -0.03, f"线上线下准确率下跌超过3%: {decay_acc:.2%}"
        t.tick("全部完成")