
import pandas as pd
import numpy as np

df = pd.read_csv(
    "daily_signal_evaluation.csv",
    index_col=0,
    parse_dates=True,
)

spread = df["top_minus_bottom"].dropna()

mean_spread = spread.mean()
std_spread = spread.std(ddof=1)
n = len(spread)

# 先计算朴素标准误和 t 统计量
# 注意：由于五日收益窗口重叠，这不是最终的有效显著性检验
standard_error = std_spread / np.sqrt(n)
t_stat = mean_spread / standard_error if standard_error > 0 else np.nan

# 用日期顺序进行区块自助法，保留部分时间相关性
rng = np.random.default_rng(42)
values = spread.to_numpy()

block_length = 5
n_bootstrap = 5000
bootstrap_means = []

for _ in range(n_bootstrap):
    sampled = []

    while len(sampled) < n:
        start = rng.integers(0, n)
        block = [
            values[(start + j) % n]
            for j in range(block_length)
        ]
        sampled.extend(block)

    bootstrap_means.append(np.mean(sampled[:n]))

ci_low, ci_high = np.percentile(
    bootstrap_means, [2.5, 97.5]
)

print("有效交易日数：", n)
print(f"平均最高组减最低组收益：{mean_spread:.6f}")
print(f"日收益差标准差：{std_spread:.6f}")
print(f"朴素 t 统计量：{t_stat:.3f}")
print(
    "5日区块自助法 95% 区间："
    f"[{ci_low:.6f}, {ci_high:.6f}]"
)

if ci_low > 0:
    print("初步结果：区间整体高于0，但仍需更严格的时间序列检验。")
elif ci_high < 0:
    print("初步结果：区间整体低于0。")
else:
    print("初步结果：区间包含0，当前证据不足以确认收益差为正。")
