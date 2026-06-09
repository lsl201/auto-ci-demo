import os
import csv
import mlflow
from datetime import datetime
import allure

# ========== 配置项（按你的环境修改） ==========
MLFLOW_TRACKING_URI = "file:///home/ubuntu/Desktop/auto-ci-demo/mlruns"
EXPERIMENT_NAME = "AI_Security_Model_Test"  # 改成你的实验名
OUTPUT_CSV_PATH = "metrics_summary.csv"

def main():
    # 1. 连接 MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found")

    # 2. 获取该实验下所有 Run
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time ASC"]
    )
    if runs.empty:
        print("No runs found")
        return

    # 3. 构建汇总表
    rows = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        start_time = datetime.fromtimestamp(run["start_time"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
        run_name = run.get("tags.mlflow.runName", run_id)

        # 读取关键指标（和你 pytest 里的 log_metric 对应）
        offline_acc = run.get("metrics.offline_accuracy", -1)
        online_acc = run.get("metrics.online_accuracy", -1)
        decay_acc = run.get("metrics.decay_accuracy", -1)
        offline_err = run.get("metrics.total_error_rate", -1)
        offline_fpr = run.get("metrics.false_positive_rate", -1)
        offline_fnr = run.get("metrics.false_negative_rate", -1)

        rows.append({
            "run_id": run_id,
            "run_name": run_name,
            "start_time": start_time,
            "offline_accuracy": round(offline_acc, 4),
            "online_accuracy": round(online_acc, 4),
            "decay_accuracy": round(decay_acc, 4),
            "offline_error_rate": round(offline_err, 4),
            "offline_fpr": round(offline_fpr, 4),
            "offline_fnr": round(offline_fnr, 4)
        })

    # 4. 写入 CSV
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ 汇总表已生成: {OUTPUT_CSV_PATH}")

    # 5. 控制台打印 Markdown 表格（方便直接看）
    print("\n📊 MLflow 全版本指标汇总：")
    print("| 版本 | 离线准确率 | 线上准确率 | 衰减 | 误判率 | FPR | FNR |")
    print("|------|------------|------------|------|--------|-----|-----|")
    for r in rows:
        print(f"| {r['run_name']} | {r['offline_accuracy']:.4f} | {r['online_accuracy']:.4f} | {r['decay_accuracy']:.4f} | {r['offline_error_rate']:.4f} | {r['offline_fpr']:.4f} | {r['offline_fnr']:.4f} |")

    # 6. 嵌入到 Allure 报告附件
    if os.path.exists(OUTPUT_CSV_PATH):
        allure.attach.file(
            source=OUTPUT_CSV_PATH,
            name="全版本指标汇总表",
            attachment_type=allure.attachment_type.CSV
        )
        print("✅ 已嵌入到 Allure 报告")

if __name__ == "__main__":
    main()