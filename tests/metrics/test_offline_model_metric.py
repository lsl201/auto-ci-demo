import allure
import pytest
import pandas as pd
import evaluate
from evaluate import load, combine
import mlflow
import requests # ✅新增：http请求本地模拟线上接口
from tests.common.model_loader import get_offline_model

# ==========1.新增：模拟线上API地址常量【改动1】==========
ONLINE_API_URL = "http://127.0.0.1:8000/predict"

EXPERIMENT = "AI_Security_Fairness_Test"
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENT)


@pytest.mark.metric
@pytest.mark.biz_metric
@allure.feature("离线模型业务指标验收+线上部署衰减对比") # ✅改动：allure描述更新
@allure.title("离线vs模拟线上API：准确率&误判率衰减核算") # ✅改动标题
def test_offline_acc_err_rate():
    """同数据集：本地离线自有模型 + 本地API(模拟线上小模型)双指标，自动计算部署衰减"""
    metric = combine([load("accuracy"),load("precision"),load("recall"),load("confusion_matrix")])
    with mlflow.start_run(run_name="离线vs模拟线上_指标衰减验收"):
        df = pd.read_csv("tests/data/offline_test_dataset.csv")
        x_data = df["input_text"].tolist()
        y_true = df["true_label"].tolist()
        mlflow.log_param("test_sample_num", len(df))

        # ==========【原有逻辑不变：离线自有模型推理】==========
        model = get_offline_model()
        y_pred_offline = model.predict(x_data) # 离线预测
        res_offline = metric.compute(predictions=y_pred_offline, references=y_true)
        tn, fp, fn, tp = res_offline["confusion_matrix"].ravel()
        # 离线指标
        acc_off = res_offline["accuracy"]
        err_off = 1 - acc_off
        fpr_off = fp/(fp+tn) if (fp+tn)!=0 else 0
        fnr_off = fn/(fn+tp) if (fn+tp)!=0 else 0
        pre_off = res_offline["precision"]
        rec_off = res_offline["recall"]

        # ==========【新增代码块：调用本地FastAPI=模拟线上推理【改动2】】==========
        try:
            resp = requests.post(ONLINE_API_URL, json={"text_list": x_data}, timeout=(3,12))
            resp.raise_for_status()
            # 下面全缩进进try内部
            y_pred_online = resp.json()["pred_label"]
            res_online = metric.compute(predictions=y_pred_online, references=y_true)
            tn_on, fp_on, fn_on, tp_on = res_online["confusion_matrix"].ravel()
            # 线上指标
            acc_on = res_online["accuracy"]
            err_on = 1 - acc_on
            fpr_on = fp_on/(fp_on+tn_on) if (fp_on+tn_on)!=0 else 0
            fnr_on = fn_on/(fn_on+tp_on) if (fn_on+tp_on)!=0 else 0
            pre_on = res_online["precision"]
            rec_on = res_online["recall"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            assert False, f"线上API{ONLINE_API_URL}未启动/连接失败：{str(e)}"

        # ==========【新增：计算指标衰减（线上-离线，负数=线上变差）【改动3】】==========
        decay_acc = acc_on - acc_off    # 准确率衰减
        decay_err = err_on - err_off    # 误判上涨幅度
        decay_fpr = fpr_on - fpr_off
        decay_fnr = fnr_on - fnr_off
        decay_pre = pre_on - pre_off
        decay_rec = rec_on - rec_off

        # ==========Allure附件：离线+线上+衰减全数据【改动4：扩充附件】==========
        all_data = {
            # 离线指标
            "off_准确率":f"{acc_off:.4f}", "off_总误判":f"{err_off:.4f}",
            "off_FPR":f"{fpr_off:.4f}", "off_FNR":f"{fnr_off:.4f}",
            "off_精确率":f"{pre_off:.4f}", "off_召回":f"{rec_off:.4f}",
            # 线上指标
            "on_准确率":f"{acc_on:.4f}", "on_总误判":f"{err_on:.4f}",
            "on_FPR":f"{fpr_on:.4f}", "on_FNR":f"{fnr_on:.4f}",
            "on_精确率":f"{pre_on:.4f}", "on_召回":f"{rec_on:.4f}",
            # 衰减值
            "准确率衰减":f"{decay_acc:.4f}", "误判涨幅":f"{decay_err:.4f}"
        }
        for k,v in all_data.items():
            allure.attach(v, name=k, attachment_type=allure.attachment_type.TEXT)

        # ==========MLflow记录：离线、线上、衰减三类指标【改动5：新增线上&衰减埋点】==========
        # 离线指标入库（原有不变）
        mlflow.log_metric("offline_accuracy", acc_off)
        mlflow.log_metric("total_error_rate", err_off)
        mlflow.log_metric("false_positive_rate_FPR", fpr_off)
        mlflow.log_metric("false_negative_rate_FNR", fnr_off)
        mlflow.log_metric("precision_off", pre_off)
        mlflow.log_metric("recall_off", rec_off)

        # ✅新增：线上版本指标入库
        mlflow.log_metric("online_accuracy", acc_on)
        mlflow.log_metric("online_total_error", err_on)
        mlflow.log_metric("online_FPR", fpr_on)
        mlflow.log_metric("online_FNR", fnr_on)
        mlflow.log_metric("precision_on", pre_on)
        mlflow.log_metric("recall_on", rec_on)

        # ✅新增：衰减差值入库，用于版本对比
        mlflow.log_metric("decay_accuracy", decay_acc)
        mlflow.log_metric("decay_err_rate", decay_err)
        mlflow.log_metric("decay_FPR", decay_fpr)
        mlflow.log_metric("decay_FNR", decay_fnr)

        # ==========业务门禁断言【改动6：新增上线衰减门禁，衰减超标阻断CI】==========
        assert acc_off >= 0.90, f"离线准确率{acc_off:.2%}<90%，基线不合格"
        assert err_off <= 0.10, f"离线误判{err_off:.2%}>10%，基线不合格"
        # 新增：线上相比离线准确率下跌不能超3%，超了判定部署失效
        assert decay_acc >= -0.03, f"线上准确率相较离线下跌{abs(decay_acc):.2%}>3%，部署衰减超标禁止上线"