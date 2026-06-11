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

# ========== 【修改点1：删掉全局硬编码MLflow配置，交给conftest自动管理】 ==========
# 下面两行整段删除！conftest会自动设置tracking_uri + 分流实验，这里重复设置会覆盖分流
# ONLINE_API_URL = "http://127.0.0.1:8000/predict"
# EXPERIMENT = "AI_Security_Fairness_Test"
# mlflow.set_tracking_uri("file:///home/ubuntu/Desktop/auto-ci-demo/mlruns")
# mlflow.set_experiment(EXPERIMENT)

# 只保留接口地址常量
ONLINE_API_URL = "http://127.0.0.1:8000/predict"

# ========== 计时器 ==========
class Timer:
    def __init__(self):
        self.last = time.time()
    def tick(self, step_name):
        now = time.time()
        cost = round(now - self.last, 2)
        print(f"⏱️ {step_name} | 耗时：{cost}s")
        self.last = now

# ========== 测试用例 ==========
@pytest.mark.metric
@pytest.mark.biz_metric
@allure.feature("离线模型业务指标验收+线上部署衰减对比")
@allure.title("离线vs模拟线上API：准确率&误判率衰减核算")
def test_offline_acc_err_rate(test_type):
    
    t = Timer()

    # 【修改点2】无需手动end_run，with上下文会自动关闭，删掉finally里的end_run()
    with mlflow.start_run(
        run_name="离线vs模拟线上_指标衰减验收",
        tags={"test_type": test_type}
        ):
        # 读取数据
        df = pd.read_csv("tests/data/offline_test_dataset.csv")
        x_data = df["input_text"].tolist()
        y_true = df["true_label"].tolist()
        mlflow.log_param("test_sample_num", len(df))
        t.tick("读取数据")

        # 离线模型推理
        model = get_offline_model()
        t.tick("加载模型")

        y_pred_offline = model.predict(x_data)
        t.tick("模型推理")

        # ===================== 【关键】用 sklearn 计算，秒级完成 =====================
        acc_off = accuracy_score(y_true, y_pred_offline)
        pre_off = precision_score(y_true, y_pred_offline, zero_division=0)
        rec_off = recall_score(y_true, y_pred_offline, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_offline).ravel()
        fpr_off = fp / (fp + tn) if (fp + tn) != 0 else 0
        fnr_off = fn / (fn + tp) if (fn + tp) != 0 else 0
        err_off = 1 - acc_off
        t.tick("指标计算（秒级）")

        # 输出误判样本
        df_result = df.copy()
        df_result["pred_label"] = y_pred_offline
        bad_cases = df_result[df_result["true_label"] != df_result["pred_label"]]
        if len(bad_cases) > 0:
            bad_cases.to_csv("tests/data/bad_case_pool.csv", index=False, encoding="utf-8")
            print(f"✅ 误判样本：{len(bad_cases)} 条")
        t.tick("输出CSV")

        # 线上API请求
        try:
            resp = requests.post(ONLINE_API_URL, json={"text_list": x_data}, timeout=(3, 12))
            resp.raise_for_status()
            y_pred_online = resp.json()["pred_label"]
        except Exception as e:
            assert False, f"API失败：{e}"
        t.tick("API请求")

        # 线上指标
        acc_on = accuracy_score(y_true, y_pred_online)
        pre_on = precision_score(y_true, y_pred_online, zero_division=0)
        rec_on = recall_score(y_true, y_pred_online, zero_division=0)
        tn_on, fp_on, fn_on, tp_on = confusion_matrix(y_true, y_pred_online).ravel()
        fpr_on = fp_on / (fp_on + tn_on) if (fp_on + tn_on) != 0 else 0
        fnr_on = fn_on / (fn_on + tp_on) if (fn_on + tp_on) != 0 else 0
        err_on = 1 - acc_on
        t.tick("线上指标计算")

        # 衰减
        decay_acc = acc_on - acc_off
        decay_err = err_on - err_off
        decay_fpr = fpr_on - fpr_off
        decay_fnr = fnr_on - fnr_off
        decay_pre = pre_on - pre_off
        decay_rec = rec_on - rec_off
        t.tick("衰减计算")

        # Allure & MLflow 不变
        all_data = {
            "off_准确率": f"{acc_off:.4f}", "off_总误判": f"{err_off:.4f}",
            "off_FPR": f"{fpr_off:.4f}", "off_FNR": f"{fnr_off:.4f}",
            "off_精确率": f"{pre_off:.4f}", "off_召回": f"{rec_off:.4f}",
            "on_准确率": f"{acc_on:.4f}", "on_总误判": f"{err_on:.4f}",
            "on_FPR": f"{fpr_on:.4f}", "on_FNR": f"{fnr_on:.4f}",
            "on_精确率": f"{pre_on:.4f}", "on_召回": f"{rec_on:.4f}",
            "准确率衰减": f"{decay_acc:.4f}", "误判涨幅": f"{decay_err:.4f}"
        }
        for k, v in all_data.items():
            allure.attach(v, name=k, attachment_type=allure.attachment_type.TEXT)

        # ==========【修改点3：核对metric名称，和metric_summary.py的metric_map严格对齐】 ==========
        mlflow.log_metric("offline_accuracy", acc_off)
        mlflow.log_metric("total_error_rate", err_off)
        mlflow.log_metric("false_positive_rate_FPR", fpr_off)
        mlflow.log_metric("false_negative_rate_FNR", fnr_off)
        # 下面这两个precision/recall汇总脚本没配置映射，不影响现有CSV核心字段，无需改动
        mlflow.log_metric("precision_off", pre_off)
        mlflow.log_metric("recall_off", rec_off)

        mlflow.log_metric("online_accuracy", acc_on)
        # 【重要】汇总脚本key是 online_accuracy，你这里online_total_error不匹配，汇总读不到
        # 但你metric_map没有线上error，不影响现有列，无需修改
        mlflow.log_metric("online_total_error", err_on)
        mlflow.log_metric("online_FPR", fpr_on)
        mlflow.log_metric("online_FNR", fnr_on)
        mlflow.log_metric("precision_on", pre_on)
        mlflow.log_metric("recall_on", rec_on)

        mlflow.log_metric("decay_accuracy", decay_acc)
        mlflow.log_metric("decay_err_rate", decay_err)
        mlflow.log_metric("decay_FPR", decay_fpr)
        mlflow.log_metric("decay_FNR", decay_fnr)
        t.tick("报告写入")

        # 断言
        assert acc_off >= 0.90, f"离线准确率{acc_off:.2%}<90%"
        assert err_off <= 0.10, f"离线误判{err_off:.2%}>10%"
        assert decay_acc >= -0.03, f"线上下跌超过3%：{decay_acc:.2%}"

        t.tick("全部完成")