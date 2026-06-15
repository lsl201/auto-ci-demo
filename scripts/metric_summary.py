import os
import csv
import mlflow
from datetime import datetime, timedelta
import sys
import pandas as pd

# =================配置区 (只改这里)================
MLFLOW_TRACKING_URI = "file:///home/ubuntu/Desktop/auto-ci-demo/mlruns/"
EXPERIMENT_NAMES = ["model-test-suite-offline", "model-test-suite-robustness"]
OUTPUT_CSV_PATH = "/home/ubuntu/Desktop/auto-ci-demo/metrics_summary.csv"

# =================工具函数================
def safe_round(v, d=4):
    try:
        return round(float(v), d)
    except (ValueError, TypeError):
        return -1

def utc_to_beijing(utc_time):
    """稳妥转北京时间"""
    try:
        if hasattr(utc_time, "tzinfo") and utc_time.tzinfo is None:
            local_time = utc_time + timedelta(hours=8)
            return local_time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(utc_time)

# =================主逻辑================
def main():
    # 1.清理旧文件
    if os.path.exists(OUTPUT_CSV_PATH):
        os.remove(OUTPUT_CSV_PATH)

    # 2.连接MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # 3.批量获取所有指定experiment的run
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

    # 按run启动时间升序: 最早执行的批次在上，最新批次在末尾
    runs = runs.sort_values(by="start_time", ascending=True, ignore_index=True)

    # 4.指标映射 (新增指标只改这里)
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

    # 5.构建行数据 (修复tags取值异常问题)
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

    # 6.写入原始CSV（原生csv模块）
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ 汇总表已生成: {OUTPUT_CSV_PATH}")

    # ==================== 新增：指标阈值门禁校验（仅告警，不阻断流水线） ====================
    # 1.读取刚生成的完整CSV
    df = pd.read_csv(OUTPUT_CSV_PATH)
    pass_flag = True

    # 2.自定义指标阈值（和你的业务对齐）
    threshold_cfg = {
        "offline_accuracy": 0.85,
        "online_accuracy": 0.83,
        "fairness_pass_rate": 0.90,
        "robust_total_max_error": 0.08
    }

    # 单行判定函数：逐条校验、生成达标标记+原因
    def judge_row(row, cfg):
        reasons = []
        # 下限类：不能低于阈值
        if row["offline_accuracy"] < cfg["offline_accuracy"]:
            reasons.append(f"离线准确率{row['offline_accuracy']:.2%}<{cfg['offline_accuracy']:.2%}")
        if "online_accuracy" in df.columns and row["online_accuracy"] < cfg["online_accuracy"]:
            reasons.append(f"线上准确率{row['online_accuracy']:.2%}<{cfg['online_accuracy']:.2%}")
        if row["fairness_pass_rate"] < cfg["fairness_pass_rate"]:
            reasons.append(f"公平性通过率{row['fairness_pass_rate']:.2%}<{cfg['fairness_pass_rate']:.2%}")

        # 鲁棒多指标均值校验
        robust_cols = ["adv_error_rate", "drift_error_rate", "ood_error_rate", "fairness_error_rate"]
        exist_cols = [c for c in robust_cols if c in df.columns]
        if len(exist_cols) > 0:
            mean_err = df.loc[row.name, exist_cols].mean()
            if mean_err > cfg["robust_total_max_error"]:
                reasons.append(f"鲁棒平均误判率{mean_err:.2%}>{cfg['robust_total_max_error']:.2%}")

        if len(reasons) == 0:
            return "达标", ""
        else:
            return "不达标", "; ".join(reasons)

    # 逐行遍历，新增两列
    df[["是否达标", "超限详情"]] = df.apply(lambda x: pd.Series(judge_row(x, threshold_cfg)), axis=1)

    # ---------------- 关键：回写CSV，新列持久化保存 ----------------
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 已追加达标校验标记，更新CSV文件")

    # 筛选不合格批次，控制台柔性告警（不exit、不阻断流水线）
    fail_df = df[df["是否达标"] == "不达标"]
    if len(fail_df) > 0:
        pass_flag = False
        print(f"\n⚠️ 柔性门禁告警：共 {len(fail_df)} 轮迭代指标不达标，已写入CSV标记，流水线继续执行！")
        print(fail_df[["run_id", "test_type", "是否达标", "超限详情"]].to_string())
    else:
        print("\n✅ 全部迭代指标校验达标")

    # 不做sys.exit(1)，永远放行流水线
    # ======================================================================================

if __name__ == "__main__":
    main()