import os
import csv
import mlflow
from datetime import datetime, timedelta
import allure
import sys
import pandas as pd

# ========== 配置区 (只改这里) ==========
MLFLOW_TRACKING_URI = "file:///home/ubuntu/Desktop/auto-ci-demo/mlruns"
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
    runs = pd.concat(all_runs, ignore_index=True)
    
    # =========新增排序代码 开始=========
    # 按run启动时间升序：最早执行的批次在上，最新批次在末尾
    runs = runs.sort_values(by="start_time", ascending=True, ignore_index=True)
    # =========新增排序代码 结束=========

    # 4. 指标映射 (新增指标只改这里)
    metric_map = {
        # 原有全局指标不动
        "offline_accuracy": "metrics.offline_accuracy",
        "total_error_rate": "metrics.total_error_rate",
        "false_positive_rate_FPR": "metrics.false_positive_rate_FPR",
        "false_negative_rate_FNR": "metrics.false_negative_rate_FNR",

        # 新增正常子集
        "offline_accuracy_normal": "metrics.offline_accuracy_normal",
        "total_error_rate_normal": "metrics.total_error_rate_normal",
        # 新增边界子集
        "offline_accuracy_edge": "metrics.offline_accuracy_edge",
        "total_error_rate_edge": "metrics.total_error_rate_edge",

        # 剩下原有online、decay、鲁棒性指标不变
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

    # 5. 构建行数据 (修复tags取值异常问题)
    rows = []
    for _, run in runs.iterrows():
        try:
            row = {
                "run_id": run["run_id"],
                "run_name": run["tags.mlflow.runName"] if "tags.mlflow.runName" in run else run["run_id"],
                "start_time": utc_to_beijing(run["start_time"]),
                "test_type": run["tags.test_type"] if "tags.test_type" in run else "unknown",
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
        print(f"✅ 总共读取了 {len(rows)} 条记录!")

    # ================== 新增: 指标阈值门禁校验 (仅告警,不阻断流水线) ==================
    # 1. 读取刚生成的完整CSV
    df = pd.read_csv(OUTPUT_CSV_PATH)
    pass_flag = True

    # 2. 自定义指标阈值 (修正key拼写错误)
    threshold_cfg = {
        "offline_accuracy": 0.85,
        "online_accuracy": 0.83,
        "fairness_pass_rate": 0.9,
        "robust_total_max_error": 0.08
    }

    # 校验离线指标 (offline分组)
    df_offline = df[df["test_type"] == "offline"]
    if len(df_offline) > 0:
        offline_acc = df_offline.iloc[0]["offline_accuracy"]
        if offline_acc < threshold_cfg["offline_accuracy"]:
            print(f"❌ 离线准确率 {offline_acc:.2%} < 阈值 {threshold_cfg['offline_accuracy']:.2%}, 指标劣化")
            pass_flag = False

    # 校验鲁棒性整体错误率均值
    df_robust = df[df["test_type"] == "robustness"]
    if len(df_robust) > 0:
        err_cols = ["adv_error_rate", "drift_error_rate", "ood_error_rate", "fairness_error_rate"]
        mean_err = df_robust[err_cols].mean(axis=1).mean()
        if mean_err > threshold_cfg["robust_total_max_error"]:
            print(f"❌ 鲁棒平均错误率 {mean_err:.4f} > 阈值 {threshold_cfg['robust_total_max_error']}")
            pass_flag = False

    # 3. 校验不达标仅打印提示, 不再exit阻断流水线 (方案B已生效)
    if not pass_flag:
        print("⚠️ 指标校验未通过，仅做提示，不阻断流水线")
    else:
        print("✅ 全部指标校验达标")


if __name__ == "__main__":
    main()