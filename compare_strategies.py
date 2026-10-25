
import pandas as pd
import numpy as np

# ==========================================
# 1. 读取两种策略的评估结果
# ==========================================

ml = pd.read_csv("daily_signal_evaluation.csv")
momentum = pd.read_csv("momentum_evaluation.csv")

# 统一日期格式
ml["date"] = pd.to_datetime(ml["date"])
momentum["date"] = pd.to_datetime(momentum["date"])

# 统一列名
ml = ml[["date", "top_minus_bottom"]].rename(
    columns={"top_minus_bottom": "ml_spread"}
)

momentum = momentum[["date", "spread"]].rename(
    columns={"spread": "momentum_spread"}
)

# ==========================================
# 2. 按日期对齐，保证比较的是同一天
# ==========================================

comparison = pd.merge(
    ml,
    momentum,
    on="date",
    how="inner",
    validate="one_to_one"
)

comparison = comparison.sort_values("date").reset_index(drop=True)

if comparison.empty:
    raise ValueError("两种策略没有相同的评估日期，请检查输入文件。")

# 机器学习策略相对动量策略的每日增量收益差
comparison["difference"] = (
    comparison["ml_spread"] - comparison["momentum_spread"]
)

# 保存逐日比较结果
comparison.to_csv("strategy_comparison.csv", index=False)

# ==========================================
# 3. 基本统计
# ==========================================

ml_spread = comparison["ml_spread"].to_numpy()
momentum_spread = comparison["momentum_spread"].to_numpy()
diff = comparison["difference"].to_numpy()

n = len(diff)

mean_ml = ml_spread.mean()
mean_momentum = momentum_spread.mean()
mean_diff = diff.mean()
std_diff = diff.std(ddof=1)

# 传统 t 统计量，仅作初步参考
if n > 1 and std_diff > 0:
    naive_t = mean_diff / (std_diff / np.sqrt(n))
else:
    naive_t = np.nan

print("=" * 50)
print("机器学习策略 vs. 传统动量策略")
print("=" * 50)

print("对齐后的交易日数：", n)
print("机器学习平均每日多空组合收益差：", round(mean_ml, 6))
print("动量策略平均每日多空组合收益差：", round(mean_momentum, 6))
print("平均每日增量收益差：", round(mean_diff, 6))
print("每日增量收益差标准差：", round(std_diff, 6))
print("普通 t 统计量（未处理时间相关性）：", round(naive_t, 4))

# ==========================================
# 4. 5日循环区块自助法
# 保留相邻交易日之间的一部分时间相关性
# ==========================================

rng = np.random.default_rng(42)

block_len = 5
n_bootstrap = 5000

if n < block_len:
    raise ValueError("有效交易日不足，无法进行区块自助法检验。")

bootstrap_means = np.empty(n_bootstrap)

for b in range(n_bootstrap):
    sampled_indices = []

    # 抽取连续5个交易日组成区块，直到样本数量足够
    while len(sampled_indices) < n:
        start = rng.integers(0, n)

        block = [
            (start + j) % n
            for j in range(block_len)
        ]

        sampled_indices.extend(block)

    sampled_indices = sampled_indices[:n]

    bootstrap_means[b] = diff[sampled_indices].mean()

ci_low, ci_high = np.percentile(
    bootstrap_means,
    [2.5, 97.5]
)

print("\n" + "=" * 50)
print("5日区块自助法检验")
print("=" * 50)

print("自助抽样次数：", n_bootstrap)
print("平均每日增量收益差：", round(mean_diff, 6))
print(
    "95%自助法区间：",
    f"[{ci_low:.6f}, {ci_high:.6f}]"
)

if ci_low > 0:
    print("初步结果：区间完全大于0，支持机器学习具有正的增量优势。")
elif ci_high < 0:
    print("初步结果：区间完全小于0，机器学习表现可能更差。")
else:
    print("初步结果：区间包含0，尚无充分证据确认机器学习具有增量优势。")

# ==========================================
# 5. 保存统计结果
# ==========================================

summary = pd.DataFrame(
    {
        "metric": [
            "aligned_days",
            "ml_mean_spread",
            "momentum_mean_spread",
            "mean_incremental_spread",
            "std_incremental_spread",
            "naive_t_statistic",
            "bootstrap_95ci_lower",
            "bootstrap_95ci_upper",
            "block_length",
            "bootstrap_repetitions",
        ],
        "value": [
            n,
            mean_ml,
            mean_momentum,
            mean_diff,
            std_diff,
            naive_t,
            ci_low,
            ci_high,
            block_len,
            n_bootstrap,
        ],
    }
)

summary.to_csv("strategy_comparison_summary.csv", index=False)

print("\n已保存逐日比较结果：strategy_comparison.csv")
print("已保存统计摘要：strategy_comparison_summary.csv")
