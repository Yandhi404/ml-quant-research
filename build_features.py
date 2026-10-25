
from pathlib import Path
import pandas as pd
import numpy as np

data_dir = Path("data")
files = sorted(
    f for f in data_dir.glob("*.csv")
    if f.stem.startswith(("sh.", "sz."))
)

feature_frames = []

for i, file in enumerate(files, start=1):
    df = pd.read_csv(file, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date")
    df = df.reset_index(drop=True)

    # 确保开盘价和收盘价是数值
    for col in ["open", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 历史收益率特征
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    # 过去20个交易日的日收益率波动率
    df["volatility_20d"] = (
        df["return_1d"].rolling(20).std()
    )

    # 当日收盘后产生信号，次日开盘买入，
    # 第5个交易日收盘卖出
    df["target_return_5d"] = (
        df["close"].shift(-5) / df["open"].shift(-1) - 1
    )

    feature_frames.append(df)

    if i % 25 == 0 or i == len(files):
        print(f"已处理 {i}/{len(files)} 只股票", flush=True)

if not feature_frames:
    raise RuntimeError("没有找到股票数据，请检查 data 文件夹。")

result = pd.concat(feature_frames, ignore_index=True)

required_cols = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "target_return_5d",
]

result = result.replace([np.inf, -np.inf], np.nan)
result = result.dropna(subset=required_cols)

output_file = Path("hs300_features.csv")
result.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n特征构建完成！")
print("股票数量：", result["code"].nunique())
print("记录数量：", len(result))
print("输出文件：", output_file.resolve())
print("\n特征预览：")
print(
    result[
        [
            "date",
            "code",
            "close",
            "return_5d",
            "return_20d",
            "volatility_20d",
            "target_return_5d",
        ]
    ].head(10).to_string(index=False)
)
