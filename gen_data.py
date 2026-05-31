import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

# 训练数据：短文本，标签均匀
train_df = pd.DataFrame({
    "text": [f"normal text sample {i}" for i in range(n_samples)],
    "target": np.random.randint(0, 2, size=n_samples)
})

# 线上数据：文本更长，标签偏斜
online_df = pd.DataFrame({
    "text": [f"very long text sample {i} with extra content " * 5 for i in range(n_samples)],
    "target": np.where(np.random.rand(n_samples) < 0.8, 0, 1)
})

train_df.to_csv("train_data.csv", index=False)
online_df.to_csv("online_data.csv", index=False)
print("✅ 数据生成完成：train_data.csv / online_data.csv")