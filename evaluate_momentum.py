
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# 读取测试集特征
df = pd.read_csv(
    "test_data.csv",
    parse_dates=["date"]
)

# 检查所需字段
required_cols = [
    "date",
    "code",
    "return_20d",
    "target_return_5d",
]

missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"数据缺少字段：{missing}")

df = df.dropna(subset=required_cols).copy()

daily_results = []

# 每个交易日单独比较股票排名
for date, group in df.groupby("date"):
    if len(group) < 10:
        continue

    # 传统动量信号：过去20个交易日涨幅
    ic, _ = spearmanr(
        group["return_20d"],
        group["target_return_5d"]
    )

    # 按动量从低到高排序，分成五组
    group = group.copy()
    group["bucket"] = pd.qcut(
        group["return_20d"].rank(method="first"),
        5,
        labels=False
    )

    lowest = group.loc[
        group["bucket"] == 0,
        "target_return_5d"
    ].mean()

    highest = group.loc[
        group["bucket"] == 4,
        "target_return_5d"
    ].mean()

    daily_results.append({
        "date": date,
        "ic": ic,
        "lowest_group_return": lowest,
        "highest_group_return": highest,
        "spread": highest - lowest,
    })

result = pd.DataFrame(daily_results)
result = result.replace([np.inf, -np.inf], np.nan)
result = result.dropna(subset=["ic", "spread"])

result.to_csv(
    "momentum_evaluation.csv",
    index=False,
    encoding="utf-8-sig"
)

print("传统20日动量策略评估")
print("有效交易日数量：", len(result))
print("平均每日 IC：", round(result["ic"].mean(), 4))
print(
    "IC 为正的交易日比例：",
    round((result["ic"] > 0).mean(), 4)
)
print(
    "平均最高组未来5日收益：",
    round(result["highest_group_return"].mean(), 6)
)
print(
    "平均最低组未来5日收益：",
    round(result["lowest_group_return"].mean(), 6)
)
print(
    "平均最高组减最低组收益：",
    round(result["spread"].mean(), 6)
)
print("\n每日结果已保存：momentum_evaluation.csv")
