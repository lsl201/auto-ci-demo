import os
import csv
import mlflow
from datetime import datetime, timedelta
import allure

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
                "test_type": run.get("tags.test_type", "unknown")  # 关键：离线/鲁棒性标签
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

if __name__ == "__main__":
    main()