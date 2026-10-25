
import pandas as pd
import numpy as np

df = pd.read_csv("all_strategies_daily.csv", parse_dates=["date"])
df = df.sort_values("date").dropna(
    subset=["ml_excess", "momentum_excess"]
)

# 每日配对差值：正数表示随机森林优于动量
diff = (
    df["ml_excess"] - df["momentum_excess"]
).to_numpy()

print("共同交易日数量：", len(diff))
print("随机森林相对动量的平均每日超额差：",
      round(diff.mean() * 100, 4), "%")
print("随机森林跑赢动量的交易日比例：",
      round((diff > 0).mean() * 100, 2), "%")

# 5日区块 Bootstrap，保留部分时间序列相关性
rng = np.random.default_rng(42)
n = len(diff)
block_size = 5
n_bootstrap = 10000
bootstrap_means = np.empty(n_bootstrap)

for i in range(n_bootstrap):
    samples = []

    while len(samples) < n:
        start = rng.integers(0, n)
        block = [
            diff[(start + j) % n]
            for j in range(block_size)
        ]
        samples.extend(block)

    bootstrap_means[i] = np.mean(samples[:n])

lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])

print("5日区块 Bootstrap 95%置信区间：")
print("[",
      round(lower * 100, 4), "%",
      ",",
      round(upper * 100, 4), "%",
      "]")

if lower > 0:
    print("初步结论：置信区间整体高于零，支持随机森林优于动量。")
elif upper < 0:
    print("初步结论：置信区间整体低于零，支持动量优于随机森林。")
else:
    print("初步结论：置信区间包含零，暂时无法确认哪种策略更优。")
