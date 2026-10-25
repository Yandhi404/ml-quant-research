import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

# ==================================================

# 1. 参数

# ==================================================

INITIAL_CAPITAL = 100_000.0

TOP_PERCENT = 0.20

HOLDING_DAYS = 5

# 单边交易成本假设：买入、卖出各 0.15%

# 这是简化假设，不代表实际券商费用

ONE_WAY_COST = 0.0015

DATA_DIR = Path("data")

PREDICTION_FILE = Path("test_predictions.csv")

RESULT_FILE = Path("backtest_v4_results.csv")

TRADE_FILE = Path("backtest_v4_trades.csv")

EQUITY_PLOT = Path("backtest_v4_equity_curve.png")

# ==================================================

# 2. 读取预测数据

# ==================================================

pred = pd.read_csv(PREDICTION_FILE)

pred["date"] = pd.to_datetime(pred["date"])

pred["code"] = pred["code"].astype(str)

required = {"date", "code", "predicted_return_5d"}

if not required.issubset(pred.columns):

    raise ValueError(f"预测文件缺少列：{required - set(pred.columns)}")

pred["predicted_return_5d"] = pd.to_numeric(

    pred["predicted_return_5d"], errors="coerce"

)

pred = pred.dropna(

    subset=["date", "code", "predicted_return_5d"]

).copy()

if pred.duplicated(["date", "code"]).any():

    raise ValueError("预测数据存在重复的日期和股票代码。")

pred = pred.sort_values(["date", "code"]).reset_index(drop=True)

if pred.empty:

    raise ValueError("预测文件没有有效数据。")

start_date = pred["date"].min()

end_date = pred["date"].max()

# ==================================================

# 3. 读取行情数据

# ==================================================

frames = []

missing_files = []

for code in pred["code"].unique():

    path = DATA_DIR / f"{code}.csv"

    if not path.exists():

        missing_files.append(code)

        continue

    df = pd.read_csv(path)

    needed = {"date", "code", "open", "close"}

    if not needed.issubset(df.columns):

        print(f"警告：{code} 缺少必要行情列，跳过。")

        continue

    df["date"] = pd.to_datetime(df["date"])

    df["code"] = df["code"].astype(str)

    df["open"] = pd.to_numeric(df["open"], errors="coerce")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    frames.append(df[["date", "code", "open", "close"]].copy())

if not frames:

    raise ValueError("没有成功读取任何股票行情。")

prices = pd.concat(frames, ignore_index=True)

if prices.duplicated(["date", "code"]).any():

    raise ValueError("行情数据存在重复的日期和股票代码。")

prices = prices.sort_values(["date", "code"]).reset_index(drop=True)

trading_dates = pd.DatetimeIndex(

    sorted(prices["date"].unique())

)

date_index = {date: i for i, date in enumerate(trading_dates)}

open_wide = prices.pivot(

    index="date", columns="code", values="open"

).reindex(trading_dates)

close_wide = prices.pivot(

    index="date", columns="code", values="close"

).reindex(trading_dates)

# 仅用于持仓日常估值，不作为实际买入成交价。

# 前向填充可能掩盖停牌或缺失行情，因此会单独记录。

close_mark = close_wide.ffill()

if missing_files:

    print(f"警告：缺少 {len(missing_files)} 个股票行情文件。")

print("预测日期范围：", start_date.date(), "至", end_date.date())

print("行情日期范围：", trading_dates.min().date(), "至", trading_dates.max().date())

# ==================================================

# 4. 生成选股信号

#

# t 日收盘后生成信号

# t+1 日开盘买入

# t+5 日收盘退出

#

# 只选择能在预测测试区间结束前完成退出的信号。

# ==================================================

signals = {}

for signal_date in sorted(pred["date"].unique()):

    idx = date_index.get(signal_date)

    if idx is None:

        continue

    entry_idx = idx + 1

    exit_idx = idx + HOLDING_DAYS

    if exit_idx >= len(trading_dates):

        continue

    entry_date = trading_dates[entry_idx]

    exit_date = trading_dates[exit_idx]

    if entry_date > end_date or exit_date > end_date:

        continue

    day = pred[pred["date"] == signal_date].copy()

    day = day.sort_values(

        "predicted_return_5d",

        ascending=False

    )

    n_select = max(1, int(np.ceil(len(day) * TOP_PERCENT)))

    signals[signal_date] = day.head(n_select)["code"].tolist()

if not signals:

    raise ValueError("没有生成有效选股信号，请检查预测日期与行情日期。")

# ==================================================

# 5. 建立五个轮换资金账户

# ==================================================

sleeve_cash = {

    i: INITIAL_CAPITAL / HOLDING_DAYS

    for i in range(HOLDING_DAYS)

}

positions = {i: None for i in range(HOLDING_DAYS)}

# pending 的每笔订单独立保存信号日期，避免日期串单

pending = {}

daily_records = []

trade_records = []

skipped_batches = 0

stale_exit_count = 0

stale_mark_count = 0

missing_open_count = 0

missing_close_count = 0

# ==================================================

# 6. 逐交易日模拟

# ==================================================

for current_date in trading_dates:

    # --------------------------------------------------

    # A. 开盘执行待买入订单

    # --------------------------------------------------

    if current_date in pending:

        (

            pending_signal_date,

            sleeve_id,

            selected_codes,

            planned_exit_date,

        ) = pending.pop(current_date)

        if positions[sleeve_id] is not None:

            raise RuntimeError(

                f"{current_date.date()}：账户 {sleeve_id} 仍有旧仓，不能开新仓。"

            )

        valid_codes = []

        for code in selected_codes:

            if code not in open_wide.columns:

                missing_open_count += 1

                continue

            px = open_wide.at[current_date, code]

            if pd.isna(px) or px <= 0:

                missing_open_count += 1

                continue

            valid_codes.append(code)

        if not valid_codes:

            skipped_batches += 1

        else:

            cash_before = sleeve_cash[sleeve_id]

            # 买入名义金额 + 买入费用不超过可用现金

            buy_notional_total = cash_before / (1 + ONE_WAY_COST)

            buy_fee_total = buy_notional_total * ONE_WAY_COST

            per_stock_notional = buy_notional_total / len(valid_codes)

            stock_positions = {}

            for code in valid_codes:

                entry_price = float(open_wide.at[current_date, code])

                quantity = per_stock_notional / entry_price

                stock_positions[code] = {

                    "entry_price": entry_price,

                    "quantity": quantity,

                    "entry_notional": per_stock_notional,

                    "buy_fee": per_stock_notional * ONE_WAY_COST,

                }

            sleeve_cash[sleeve_id] = (

                cash_before - buy_notional_total - buy_fee_total

            )

            # 理论上只剩浮点误差

            if abs(sleeve_cash[sleeve_id]) < 1e-8:

                sleeve_cash[sleeve_id] = 0.0

            positions[sleeve_id] = {

                "signal_date": pending_signal_date,

                "entry_date": current_date,

                "exit_date": planned_exit_date,

                "stocks": stock_positions,

            }

    # --------------------------------------------------

    # B. 收盘时退出到期持仓

    # --------------------------------------------------

    for sleeve_id in range(HOLDING_DAYS):

        position = positions[sleeve_id]

        if position is None:

            continue

        if current_date < position["exit_date"]:

            continue

        proceeds_total = 0.0

        batch_exit_details = []

        for code, stock in position["stocks"].items():

            exit_price = np.nan

            exit_price_source = "official_close"

            if code in close_wide.columns:

                exit_price = close_wide.at[current_date, code]

            # 不用未来价格补缺失成交价。

            # 如果当天没有有效收盘价，使用最后可见收盘价做估值，

            # 并明确标记为估算退出，不能视为真实可成交结果。

            if pd.isna(exit_price) or exit_price <= 0:

                exit_price = (

                    close_mark.at[current_date, code]

                    if code in close_mark.columns

                    else np.nan

                )

                exit_price_source = "stale_close_estimate"

                stale_exit_count += 1

            if pd.isna(exit_price) or exit_price <= 0:

                raise ValueError(

                    f"{current_date.date()} {code} 无法估值退出，"

                    "请检查行情数据。"

                )

            exit_price = float(exit_price)

            quantity = stock["quantity"]

            exit_notional = quantity * exit_price

            sell_fee = exit_notional * ONE_WAY_COST

            net_pnl = (

                exit_notional

                - sell_fee

                - stock["entry_notional"]

                - stock["buy_fee"]

            )

            invested_with_fee = (

                stock["entry_notional"] + stock["buy_fee"]

            )

            net_return = net_pnl / invested_with_fee

            holding_days = (

                date_index[current_date]

                - date_index[position["entry_date"]]

                + 1

            )

            batch_exit_details.append({

                "signal_date": position["signal_date"],

                "entry_date": position["entry_date"],

                "exit_date": current_date,

                "sleeve_id": sleeve_id,

                "code": code,

                "entry_price": stock["entry_price"],

                "exit_price": exit_price,

                "quantity": quantity,

                "entry_notional": stock["entry_notional"],

                "exit_notional": exit_notional,

                "buy_fee": stock["buy_fee"],

                "sell_fee": sell_fee,

                "net_pnl": net_pnl,

                "net_return": net_return,

                "holding_trading_days": holding_days,

                "exit_price_source": exit_price_source,

            })

            proceeds_total += exit_notional - sell_fee

        sleeve_cash[sleeve_id] += proceeds_total

        trade_records.extend(batch_exit_details)

        positions[sleeve_id] = None

    # --------------------------------------------------

    # C. 收盘计算组合资产

    # --------------------------------------------------

    total_cash = sum(sleeve_cash.values())

    invested_value = 0.0

    stale_mark_today = 0

    for sleeve_id in range(HOLDING_DAYS):

        position = positions[sleeve_id]

        if position is None:

            continue

        for code, stock in position["stocks"].items():

            mark_price = np.nan

            if code in close_wide.columns:

                mark_price = close_wide.at[current_date, code]

            if pd.isna(mark_price) or mark_price <= 0:

                mark_price = (

                    close_mark.at[current_date, code]

                    if code in close_mark.columns

                    else np.nan

                )

                stale_mark_today += 1

            if pd.notna(mark_price) and mark_price > 0:

                invested_value += (

                    stock["quantity"] * float(mark_price)

                )

            else:

                missing_close_count += 1

    stale_mark_count += stale_mark_today

    total_equity = total_cash + invested_value

    daily_records.append({

        "date": current_date,

        "cash": total_cash,

        "invested_value": invested_value,

        "equity": total_equity,

        "stale_marks": stale_mark_today,

    })

    # --------------------------------------------------

    # D. 收盘后产生新信号，安排次日开仓

    # --------------------------------------------------

    if current_date not in signals:

        continue

    idx = date_index[current_date]

    entry_idx = idx + 1

    exit_idx = idx + HOLDING_DAYS

    if exit_idx >= len(trading_dates):

        continue

    entry_date = trading_dates[entry_idx]

    exit_date = trading_dates[exit_idx]

    # 每五个全局交易日轮换一个账户。

    # 该逻辑假定信号每日连续产生；遇到缺失信号时可能出现账户空档。

    sleeve_id = idx % HOLDING_DAYS

    if entry_date in pending:

        raise RuntimeError(

            f"{entry_date.date()} 出现重复待执行订单。"

        )

    if positions[sleeve_id] is not None:

        raise RuntimeError(

            f"{current_date.date()}：轮换到的账户 {sleeve_id} 仍有持仓。"

        )

    pending[entry_date] = (

        current_date,

        sleeve_id,

        signals[current_date],

        exit_date,

    )

# ==================================================

# 7. 输出每日净值

# ==================================================

results = pd.DataFrame(daily_records)

results = results[

    (results["date"] >= start_date)

    & (results["date"] <= end_date)

].copy()

results = results.sort_values("date").reset_index(drop=True)

if results.empty:

    raise ValueError("回测结果为空。")

results["net_daily_return"] = results["equity"].pct_change()

results.loc[0, "net_daily_return"] = (

    results.loc[0, "equity"] / INITIAL_CAPITAL - 1

)

results["drawdown"] = (

    results["equity"] / results["equity"].cummax() - 1

)

total_return = results["equity"].iloc[-1] / INITIAL_CAPITAL - 1

max_drawdown = results["drawdown"].min()

daily_returns = results["net_daily_return"].dropna()

if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0:

    sharpe = (

        daily_returns.mean()

        / daily_returns.std(ddof=1)

        * np.sqrt(252)

    )

else:

    sharpe = np.nan

results.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")

trades = pd.DataFrame(trade_records)

if not trades.empty:

    trades.to_csv(TRADE_FILE, index=False, encoding="utf-8-sig")

else:

    pd.DataFrame(columns=[

        "signal_date", "entry_date", "exit_date", "sleeve_id",

        "code", "entry_price", "exit_price", "quantity",

        "entry_notional", "exit_notional", "buy_fee", "sell_fee",

        "net_pnl", "net_return", "holding_trading_days",

        "exit_price_source"

    ]).to_csv(TRADE_FILE, index=False, encoding="utf-8-sig")

# ==================================================

# 8. 资金对账

# ==================================================

if not trades.empty:

    trade_net_pnl = trades["net_pnl"].sum()

else:

    trade_net_pnl = 0.0

equity_change = results["equity"].iloc[-1] - INITIAL_CAPITAL

reconciliation_difference = equity_change - trade_net_pnl

# 只有所有持仓都退出、没有剩余投资市值时，

# 期末权益变化才应与逐笔已实现净盈亏直接一致。

final_invested_value = results["invested_value"].iloc[-1]

all_positions_closed = all(

    positions[sleeve_id] is None

    for sleeve_id in range(HOLDING_DAYS)

)

print("=" * 60)

print("可审计每日盯市回测 v4")

print("=" * 60)

print("测试开始日期：", results["date"].min().date())

print("测试结束日期：", results["date"].max().date())

print("回测交易日数：", len(results))

print("初始资金：", f"{INITIAL_CAPITAL:,.2f}")

print("期末权益：", f"{results['equity'].iloc[-1]:,.2f}")

print("期末现金：", f"{results['cash'].iloc[-1]:,.2f}")

print("期末持仓估值：", f"{final_invested_value:,.2f}")

print("累计收益率：", f"{total_return:.2%}")

print("最大回撤：", f"{max_drawdown:.2%}")

print("粗略年化夏普比率：", round(sharpe, 3))

print("单边成本假设：", f"{ONE_WAY_COST:.2%}")

print("逐笔交易记录数：", len(trades))

print("跳过的开仓批次：", skipped_batches)

print("缺失开盘价的股票次数：", missing_open_count)

print("使用滞后收盘价估算退出的次数：", stale_exit_count)

print("使用滞后价格估值的持仓次数：", stale_mark_count)

print("无法估值的持仓次数：", missing_close_count)

print("逐笔净盈亏合计：", f"{trade_net_pnl:,.2f}")

print("期末权益变化：", f"{equity_change:,.2f}")

print("对账差额（权益变化 - 逐笔净盈亏）：",

      f"{reconciliation_difference:,.6f}")

print("所有账户期末均空仓：", all_positions_closed)

if all_positions_closed and abs(reconciliation_difference) > 0.05:

    print("警告：期末账户均空仓，但逐笔交易与期末权益未能对账。")

if not all_positions_closed:

    print("提示：期末仍有持仓，权益变化不应直接等于已实现净盈亏。")

if not trades.empty:

    print(

        "持有交易日统计：",

        trades["holding_trading_days"].value_counts().sort_index().to_dict()

    )

print("每日净值文件：", RESULT_FILE)

print("逐笔交易文件：", TRADE_FILE)

# ==================================================

# 9. 绘制净值曲线

# ==================================================

plt.figure(figsize=(12, 6))

plt.plot(

    results["date"],

    results["equity"] / INITIAL_CAPITAL,

    label="ML Strategy v4",

)

plt.axhline(

    1.0,

    linestyle="--",

    label="Initial Capital",

)

plt.title("ML Strategy Equity Curve (v4)")

plt.xlabel("Date")

plt.ylabel("Portfolio Value / Initial Capital")

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(EQUITY_PLOT, dpi=150)

plt.close()

print("净值曲线文件：", EQUITY_PLOT)

print("回测运行完成。")