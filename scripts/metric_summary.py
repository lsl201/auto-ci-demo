import os
import csv
import mlflow
from datetime import datetime, timedelta
import allure
import sys
import pandas as pd


# ========== 配置区（只改这里） ==========
MLFLOW_TRACKING_URI = "file:///home/ubuntu/Desktop/auto-ci-demo/mlruns"
# 支持多个experiment，未来拆成 offline/robustness 直接加进来
# 原来：EXPERIMENT_NAMES = ["model-test-suite"]
EXPERIMENT_NAMES = ["model-test-suite-offline", "model-test-suite-robustness"]
OUTPUT_CSV_PATH = "/home/ubuntu/Desktop/auto-ci-demo/metrics_summary.csv"

# ========== 工具函数 ==========
def safe_round(v, d=4):
    try:
        return round(float(v), d)
    except (ValueError, TypeError):
        return -1

def utc_to_beijing(utc_time):
    """稳妥转北京时间"""
    try:
        if hasattr(utc_time, "tzinfo") and utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=None)
        local_time = utc_time + timedelta(hours=8)
        return local_time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(utc_time)

# ========== 主逻辑 ==========
def main():
    # 1. 清理旧文件
    if os.path.exists(OUTPUT_CSV_PATH):
        os.remove(OUTPUT_CSV_PATH)

    # 2. 连接MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # 3. 批量获取所有指定experiment的run
    all_runs = []
    for exp_name in EXPERIMENT_NAMES:
        exp = mlflow.get_experiment_by_name(exp_name)
        if not exp:
            print(f"⚠️ Experiment '{exp_name}' 不存在，跳过")
            continue
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time ASC"]
        )
        if not runs.empty:
            all_runs.append(runs)

    if not all_runs:
        print("⚠️ 没有找到任何运行记录")
        return

    # 合并所有experiment的数据
    import pandas as pd
    runs = pd.concat(all_runs, ignore_index=True)

    # 4. 指标映射（新增指标只改这里）
    metric_map = {
        "offline_accuracy": "metrics.offline_accuracy",
        "online_accuracy": "metrics.online_accuracy",
        "decay_accuracy": "metrics.decay_accuracy",
        "offline_error_rate": "metrics.total_error_rate",
        "offline_fpr": "metrics.false_positive_rate_FPR",
        "offline_fnr": "metrics.false_negative_rate_FNR",
        "adv_robust_rate": "metrics.adv_robust_rate",
        "adv_error_rate": "metrics.adv_error_rate",
        "drift_robust_rate": "metrics.drift_robust_rate",
        "drift_error_rate": "metrics.drift_error_rate",
        "ood_error_rate": "metrics.ood_error_rate",
        "ood_wrong_count": "metrics.ood_wrong_count",
        "noise_robust_accuracy": "metrics.robust_noise_accuracy",
        "noise_robust_error_rate": "metrics.robust_noise_error_rate",
        "fairness_pass_rate": "metrics.fairness_pass_rate",
        "fairness_error_rate": "metrics.fairness_error_rate"
    }

    # 5. 构建行数据
    rows = []
    for _, run in runs.iterrows():
        try:
            row = {
                    "run_id": run["run_id"],
                    "run_name": run.get("tags.mlflow.runName", run["run_id"]),
                    "start_time": utc_to_beijing(run["start_time"]),
                    "test_type": run.get("tags.test_type", "unknown"),  # 关键：离线/鲁棒性标签
                    # ========== 新增模型版本字段 ==========
                    "model_version": run.get("params.model_version", "unknown"),
                    "model_image": run.get("params.model_image", "unknown"),
                    "weight_file_md5": run.get("params.weight_file_md5", "unknown"),
            }
            # 批量填充指标
            for col, key in metric_map.items():
                row[col] = safe_round(run.get(key, -1))
            rows.append(row)
        except Exception as e:
            print(f"❌ 处理run {run['run_id']} 失败: {e}")
            continue

    # 6. 写入CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ 汇总表已生成: {OUTPUT_CSV_PATH}")
        print(f"✅ 总共读取了 {len(rows)} 条记录！")
    # ====================== 新增：指标阈值门禁校验 ======================
    # 1. 读取刚生成的完整CSV
    df = pd.read_csv(OUTPUT_CSV_PATH)
    pass_flag = True

    # 2. 自定义指标阈值（按需修改）
    threshold_cfg = {
        "offline_accuracy": 0.85,
        "online_accuracy": 0.83,
        "fairness_pass_rate": 0.9,
        "robust_total_max_error": 0.08
    }

    # 校验离线指标（offline分组）
    df_offline = df[df["test_type"] == "offline"]
    if len(df_offline) > 0:
        offline_acc = df_offline.iloc[0]["offline_accuracy"]
        if offline_acc < threshold_cfg["offline_accuracy"]:
            print(f"❌ 离线准确率 {offline_acc} < 阈值 {threshold_cfg['offline_accuracy']}，指标劣化")
            pass_flag = False

    # 校验鲁棒性整体错误率均值
    df_robust = df[df["test_type"] == "robustness"]
    if len(df_robust) > 0:
        mean_err = df_robust[["adv_error_rate","drift_error_rate","ood_error_rate","fairness_error_rate"]].mean(axis=1).mean()
        if mean_err > threshold_cfg["robust_total_max_error"]:
            print(f"❌ 鲁棒平均错误率 {mean_err:.4f} > 阈值 {threshold_cfg['robust_total_max_error']}")
            pass_flag = False

    # 3. 校验不达标，直接退出，不执行模型注册
    if not pass_flag:
        print("❌ 指标校验未通过，终止流水线，不注册模型版本")
        sys.exit(1)

    print("✅ 全部指标校验达标，开始执行模型版本注册流程")
    # ==================================================================
    # ====================== MLflow模型注册&环境流转 ======================
    model_version = os.getenv("MODEL_VERSION")
    if not model_version:
        print("⚠️ 未获取到MODEL_VERSION环境变量，跳过模型注册")
        sys.exit(0)

    exp_name_offline = "model-test-suite-offline"
    try:
        exp_obj = mlflow.get_experiment_by_name(exp_name_offline)
        if not exp_obj:
            print(f"⚠️ 实验 {exp_name_offline} 不存在，跳过注册")
            sys.exit(0)

        # 取本次最新一条离线Run
        latest_runs = mlflow.search_runs(
            experiment_ids=[exp_obj.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        if latest_runs.empty:
            print("⚠️ 离线实验无可用Run，跳过注册")
            sys.exit(0)

        latest_run_id = latest_runs.iloc[0]["run_id"]
        model_name = "chat-llm-service"

        # 注册模型到Model Registry
        reg_result = mlflow.register_model(
            model_uri=f"runs:/{latest_run_id}/model",
            name=model_name,
            tags={"model_version": model_version,
                  "allure_report_url": "你的Jenkins对应构建的Allure外网访问地址"
                  }
        )
        print(f"✅ 模型版本 {model_version} 注册成功，MLflow模型版本号：{reg_result.version}")

        # 流转到Staging测试环境
        client = mlflow.tracking.MlflowClient()
        ver_num = model_version.lstrip("v")
        client.transition_model_version_stage(
            name=model_name,
            version=ver_num,
            stage="Staging",
            archive_existing_versions=False
        )
        print(f"✅ 版本 {model_version} 已自动划入Staging测试环境，等待人工审批上线")

    except Exception as e:
        print(f"⚠️ 模型注册流程异常，但测试指标已达标，不阻断流水线：{str(e)}")
        

if __name__ == "__main__":
    main()