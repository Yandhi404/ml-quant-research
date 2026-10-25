
import pandas as pd

df = pd.read_csv(
    "test_predictions.csv",
    parse_dates=["date"],
)

pred_col = "predicted_return_5d"
target_col = "target_return_5d"

# 每个交易日单独计算横截面排名
def evaluate_day(group):
    group = group.dropna(subset=[pred_col, target_col]).copy()

    if len(group) < 20:
        return None

    # 预测排名与真实收益排名的 Spearman 相关系数
    ic = group[pred_col].corr(
        group[target_col],
        method="spearman",
    )

    # 每天按预测值排名，分成五组
    rank = group[pred_col].rank(method="first", pct=True)
    group["bucket"] = (rank * 5).astype(int).clip(upper=4)

    top_return = group.loc[
        group["bucket"] == 4, target_col
    ].mean()

    bottom_return = group.loc[
        group["bucket"] == 0, target_col
    ].mean()

    return pd.Series({
        "IC": ic,
        "top_return": top_return,
        "bottom_return": bottom_return,
        "top_minus_bottom": top_return - bottom_return,
        "stock_count": len(group),
    })

daily = (
    df.groupby("date")
    .apply(evaluate_day)
    .dropna()
)

print("测试交易日数量：", len(daily))
print("平均每日 IC：", round(daily["IC"].mean(), 4))
print("IC 为正的交易日比例：", round((daily["IC"] > 0).mean(), 4))
print(
    "平均最高组未来5日收益：",
    round(daily["top_return"].mean(), 4),
)
print(
    "平均最低组未来5日收益：",
    round(daily["bottom_return"].mean(), 4),
)
print(
    "平均最高组减最低组收益：",
    round(daily["top_minus_bottom"].mean(), 4),
)

daily.to_csv("daily_signal_evaluation.csv", encoding="utf-8-sig")
print("\n每日评估结果已保存：daily_signal_evaluation.csv")
