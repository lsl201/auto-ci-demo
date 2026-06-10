import os
import csv
import mlflow
from datetime import datetime
import allure

# ========== 配置项（已修复） ==========
MLFLOW_TRACKING_URI = "file:///home/ubuntu/Desktop/auto-ci-demo/mlruns"
EXPERIMENT_NAME = "model-test-suite"  # 👈 改成你真实的实验名
OUTPUT_CSV_PATH = "/home/ubuntu/Desktop/auto-ci-demo/metrics_summary.csv"  # 👈 改这里【绝对路径】

# ========== 【关键修复1】强制删除旧CSV，避免脏数据 ==========
if os.path.exists(OUTPUT_CSV_PATH):
    os.remove(OUTPUT_CSV_PATH)

def safe_round(v, d=4):
    try:
        return round(float(v), d)
    except:
        return -1

def main():
    # 1. 连接 MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' 不存在")
        return

    # 2. 获取所有 Run
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time ASC"]
    )
    if runs.empty:
        print("⚠️ 没有找到运行记录")
        return

    # 3. 构建汇总表
    rows = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]

        # 时间格式化修复（UTC → 北京时间 UTC+8）
        try:
            start_time = run["start_time"]
            from datetime import timedelta
            local_time = start_time + timedelta(hours=8)
            start_time = local_time.strftime("%Y-%m-%d %H:%M:%S")
        except:
            start_time = str(run["start_time"])

        run_name = run.get("tags.mlflow.runName", run_id)

        # 离线vs模拟线上指标
        offline_acc = run.get("metrics.offline_accuracy", -1)
        online_acc = run.get("metrics.online_accuracy", -1)
        decay_acc = run.get("metrics.decay_accuracy", -1)
        offline_err = run.get("metrics.total_error_rate", -1)
        offline_fpr = run.get("metrics.false_positive_rate_FPR", -1)
        offline_fnr = run.get("metrics.false_negative_rate_FNR", -1)

        # 鲁棒性测试指标-对抗注入指标
        adv_robust = run.get("metrics.adv_robust_rate", -1)
        adv_error = run.get("metrics.adv_error_rate", -1)

        # 鲁棒性测试指标-数据漂移指标
        drift_robust = run.get("metrics.drift_robust_rate", -1)
        drift_error = run.get("metrics.drift_error_rate", -1)

        # 鲁棒性测试指标-OOD测试指标
        ood_error_rate = run.get("metrics.ood_error_rate", -1)
        ood_wrong_count = run.get("metrics.ood_wrong_count", -1)

        # 鲁棒性测试指标-脏输入测试指标
        noise_robust_accuracy = run.get("metrics.robust_noise_accuracy", -1)
        noise_robust_error = run.get("metrics.robust_noise_error_rate", -1)

        # 鲁棒性测试指标-公平性测试指标
        fairness_pass_rate = run.get("metrics.fairness_pass_rate", -1)
        fairness_error_rate = run.get("metrics.fairness_error_rate", -1)

        # 安全四舍五入
        def safe_round(v, d=4):
            try:
                return round(float(v), d)
            except:
                return -1

        rows.append({
            "run_id": run_id,
            "run_name": run_name,
            "start_time": start_time,
            "offline_accuracy": safe_round(offline_acc),
            "online_accuracy": safe_round(online_acc),
            "decay_accuracy": safe_round(decay_acc),
            "offline_error_rate": safe_round(offline_err),
            "offline_fpr": safe_round(offline_fpr),
            "offline_fnr": safe_round(offline_fnr),
            "adv_robust_rate": safe_round(adv_robust),
            "adv_error_rate": safe_round(adv_error),
            "drift_robust_rate": safe_round(drift_robust),
            "drift_error_rate": safe_round(drift_error),
            "ood_error_rate": safe_round(ood_error_rate),
            "ood_wrong_count": safe_round(ood_wrong_count),
            "noise_robust_accuracy": safe_round(noise_robust_accuracy),
            "noise_robust_error_rate": safe_round(noise_robust_error),
            "fairness_pass_rate": safe_round(fairness_pass_rate),
            "fairness_error_rate": safe_round(fairness_error_rate)
        })

    # 4. 写入 CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:  # 👈 模式 w = 覆盖
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ 汇总表已生成: {OUTPUT_CSV_PATH}")
        print(f"✅ 总共读取了 {len(rows)} 条记录！")

if __name__ == "__main__":
    main()