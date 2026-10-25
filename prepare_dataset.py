
import pandas as pd

df = pd.read_csv(
    "hs300_features.csv",
    parse_dates=["date"],
)

df = df.sort_values(["date", "code"]).reset_index(drop=True)

feature_cols = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
]

target_col = "target_return_5d"

# 按日期划分，避免随机拆分时间序列
dates = sorted(df["date"].dropna().unique())

train_end = pd.Timestamp("2024-12-31")
valid_end = pd.Timestamp("2025-12-31")

train = df[df["date"] <= train_end].copy()
valid = df[
    (df["date"] > train_end) &
    (df["date"] <= valid_end)
].copy()
test = df[df["date"] > valid_end].copy()

print("数据集时间范围：")
print("全部数据：", df["date"].min().date(), "至", df["date"].max().date())

for name, part in [
    ("训练集", train),
    ("验证集", valid),
    ("测试集", test),
]:
    print(f"\n{name}")
    print("记录数：", len(part))
    if not part.empty:
        print("日期：", part["date"].min().date(), "至", part["date"].max().date())
        print("股票数：", part["code"].nunique())
        print("目标收益率均值：", round(part[target_col].mean(), 6))

# 保存拆分结果，供下一步训练使用
train.to_csv("train_data.csv", index=False, encoding="utf-8-sig")
valid.to_csv("valid_data.csv", index=False, encoding="utf-8-sig")
test.to_csv("test_data.csv", index=False, encoding="utf-8-sig")

print("\n训练集、验证集和测试集已保存。")
