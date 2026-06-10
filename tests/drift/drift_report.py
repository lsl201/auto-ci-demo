from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import pandas as pd

# 读本地数据
train_df = pd.read_csv("train_data.csv")
online_df = pd.read_csv("online_data.csv")

# 初始化漂移报告
report = Report(metrics=[
    DataDriftPreset(),   # 数据漂移
    TargetDriftPreset()  # 标签漂移
])

# 运行对比
report.run(reference_data=train_df, current_data=online_df)

# 保存 HTML 报告
report.save_html("data_drift_report.html")
print("✅ 漂移报告已生成：data_drift_report.html")