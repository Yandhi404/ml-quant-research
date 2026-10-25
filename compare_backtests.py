import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

# =====================================================

# 1. 统一参数

# =====================================================

INITIAL_CAPITAL = 100_000.0

TOP_PERCENT = 0.20

HOLDING_DAYS = 5

ONE_WAY_COST = 0.0015

RANDOM_SEED = 42

DATA_DIR = Path("data")

PREDICTION_FILE = Path("test_predictions.csv")

TEST_DATA_FILE = Path("test_data.csv")

RESULT_FILE = Path("strategy_comparison_backtests.csv")

SUMMARY_FILE = Path("strategy_comparison_backtests_summary.csv")

TRADE_FILE = Path("strategy_comparison_backtests_trades.csv")

PLOT_FILE = Path("strategy_comparison_backtests.png")

STRATEGY_NAMES = ["RandomForest", "Momentum20D", "Random"]

# =====================================================

# 2. 读取数据，构造共同股票池

# =====================================================

ml = pd.read_csv(PREDICTION_FILE, parse_dates=["date"])

test = pd.read_csv(TEST_DATA_FILE, parse_dates=["date"])

ml["code"] = ml["code"].astype(str)

test["code"] = test["code"].astype(str)

required_ml = {"date", "code", "predicted_return_5d"}

required_test = {"date", "code", "return_20d"}

if not required_ml.issubset(ml.columns):

    raise ValueError(f"预测文件缺少列：{required_ml - set(ml.columns)}")

if not required_test.issubset(test.columns):

    raise ValueError(f"测试文件缺少列：{required_test - set(test.columns)}")

ml["predicted_return_5d"] = pd.to_numeric(

    ml["predicted_return_5d"], errors="coerce"

)

test["return_20d"] = pd.to_numeric(

    test["return_20d"], errors="coerce"

)

ml = ml.dropna(

    subset=["date", "code", "predicted_return_5d"]

)

test = test.dropna(

    subset=["date", "code", "return_20d"]

)

if ml.duplicated(["date", "code"]).any():

    raise ValueError("预测文件存在重复的日期和股票。")

if test.duplicated(["date", "code"]).any():

    raise ValueError("测试文件存在重复的日期和股票。")

data = ml[

    ["date", "code", "predicted_return_5d"]

].merge(

    test[["date", "code", "return_20d"]],

    on=["date", "code"],

    how="inner",

)

if data.empty:

    raise ValueError("预测数据与测试数据没有共同的日期和股票。")

# =====================================================

# 3. 读取实际行情

# =====================================================

frames = []

missing_files = []

for code in data["code"].unique():

    path = DATA_DIR / f"{code}.csv"

    if not path.exists():

        missing_files.append(code)

        continue

    df = pd.read_csv(path)

    required_price = {"date", "code", "open", "close"}

    if not required_price.issubset(df.columns):

        print(f"警告：{code} 行情列不完整，已跳过。")

        continue

    df["date"] = pd.to_datetime(df["date"])

    df["code"] = df["code"].astype(str)

    df["open"] = pd.to_numeric(df["open"], errors="coerce")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    frames.append(df[["date", "code", "open", "close"]])

if not frames:

    raise ValueError("没有可用的行情数据。")

prices = pd.concat(frames, ignore_index=True)

if prices.duplicated(["date", "code"]).any():

    raise ValueError("行情数据存在重复的日期和股票。")

# 仅保留有行情文件的共同股票

available_codes = set(prices["code"].unique())

data = data[data["code"].isin(available_codes)].copy()

if data.empty:

    raise ValueError("共同股票池中没有可用行情。")

prices = prices.sort_values(["date", "code"])

trading_dates = pd.DatetimeIndex(

    sorted(prices["date"].unique())

)

trading_dates = pd.DatetimeIndex(

    [pd.Timestamp(d) for d in trading_dates]

)

date_index = {

    date: i for i, date in enumerate(trading_dates)

}

open_wide = prices.pivot(

    index="date", columns="code", values="open"

).reindex(trading_dates)

close_wide = prices.pivot(

    index="date", columns="code", values="close"

).reindex(trading_dates)

# 前向填充只用于标记缺失成交价格时的估算值

close_mark = close_wide.ffill()

start_date = pd.Timestamp(data["date"].min())

end_date = pd.Timestamp(data["date"].max())

print("=" * 65)

print("数据检查")

print("=" * 65)

print("共同测试日期：", start_date.date(), "至", end_date.date())

print("共同股票数量：", data["code"].nunique())

print("共同测试记录：", len(data))

print("缺少行情文件数量：", len(missing_files))

# =====================================================

# 4. 生成三种策略的选股信号

# =====================================================

signals = {name: {} for name in STRATEGY_NAMES}

rng = np.random.default_rng(RANDOM_SEED)

for signal_date, group in data.groupby("date", sort=True):

    signal_date = pd.Timestamp(signal_date)

    idx = date_index.get(signal_date)

    if idx is None:

        continue

    entry_idx = idx + 1

    exit_idx = idx + HOLDING_DAYS

    # 关键修正：

    # 必须能在共同测试区间结束前完成退出

    if exit_idx >= len(trading_dates):

        continue

    entry_date = trading_dates[entry_idx]

    exit_date = trading_dates[exit_idx]

    if entry_date > end_date or exit_date > end_date:

        continue

    group = group.copy()

    n_select = max(

        1,

        int(np.ceil(len(group) * TOP_PERCENT))

    )

    # 机器学习预测排序

    ml_group = group.sort_values(

        "predicted_return_5d",

        ascending=False,

        kind="mergesort",

    )

    # 传统20日动量排序

    mom_group = group.sort_values(

        "return_20d",

        ascending=False,

        kind="mergesort",

    )

    signals["RandomForest"][signal_date] = (

        ml_group.head(n_select)["code"].tolist()

    )

    signals["Momentum20D"][signal_date] = (

        mom_group.head(n_select)["code"].tolist()

    )

    # 固定随机种子，保证结果可复现

    signals["Random"][signal_date] = rng.choice(

        group["code"].to_numpy(),

        size=min(n_select, len(group)),

        replace=False,

    ).tolist()

if not any(signals.values()):

    raise ValueError("没有可以完整退出的交易信号。")

print("可完整退出的信号日期数：", len(signals["RandomForest"]))

# =====================================================

# 5. 通用组合回测引擎

# ====================================================
def run_backtest(signal_map, strategy_name):

    sleeve_cash = {

        i: INITIAL_CAPITAL / HOLDING_DAYS

        for i in range(HOLDING_DAYS)

    }

    positions = {i: None for i in range(HOLDING_DAYS)}

    pending = {}

    daily_records = []

    trade_records = []

    skipped_batches = 0

    missing_open_count = 0

    estimated_exit_count = 0

    estimated_mark_count = 0

    for current_date in trading_dates:

        # -------------------------------------------------

        # A. 开盘买入

        # -------------------------------------------------

        if current_date in pending:

            signal_date, sleeve_id, codes, exit_date = (

                pending.pop(current_date)

            )

            if positions[sleeve_id] is not None:

                raise RuntimeError(

                    f"{strategy_name}: 账户仍有旧仓，无法开仓。"

                )

            valid_codes = []

            for code in codes:

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

                cash = sleeve_cash[sleeve_id]

                # 买入金额与买入费用合计不超过可用现金

                total_notional = cash / (1 + ONE_WAY_COST)

                total_buy_fee = total_notional * ONE_WAY_COST

                per_stock = total_notional / len(valid_codes)

                stocks = {}

                for code in valid_codes:

                    entry_price = float(

                        open_wide.at[current_date, code]

                    )

                    quantity = per_stock / entry_price

                    stocks[code] = {

                        "entry_price": entry_price,

                        "quantity": quantity,

                        "entry_notional": per_stock,

                        "buy_fee": per_stock * ONE_WAY_COST,

                    }

                sleeve_cash[sleeve_id] = (

                    cash - total_notional - total_buy_fee

                )

                if abs(sleeve_cash[sleeve_id]) < 1e-8:

                    sleeve_cash[sleeve_id] = 0.0

                positions[sleeve_id] = {

                    "signal_date": signal_date,

                    "entry_date": current_date,

                    "exit_date": exit_date,

                    "stocks": stocks,

                }

        # -------------------------------------------------

        # B. 到期后收盘卖出

        # -------------------------------------------------

        for sleeve_id in range(HOLDING_DAYS):

            pos = positions[sleeve_id]

            if pos is None or current_date < pos["exit_date"]:

                continue

            proceeds_total = 0.0

            for code, stock in pos["stocks"].items():

                exit_price = np.nan

                price_source = "official_close"

                if code in close_wide.columns:

                    exit_price = close_wide.at[

                        current_date, code

                    ]

                # 没有有效收盘价时，使用最近可见价格估算。

                # 该交易会被明确标记，不能视为真实成交。

                if pd.isna(exit_price) or exit_price <= 0:

                    price_source = "stale_close_estimate"

                    if code in close_mark.columns:

                        exit_price = close_mark.at[

                            current_date, code

                        ]

                    estimated_exit_count += 1

                if pd.isna(exit_price) or exit_price <= 0:

                    raise ValueError(

                        f"{strategy_name}: {current_date.date()} "

                        f"{code} 无法估值退出。"

                    )

                exit_price = float(exit_price)

                exit_notional = stock["quantity"] * exit_price

                sell_fee = exit_notional * ONE_WAY_COST

                net_pnl = (

                    exit_notional

                    - sell_fee

                    - stock["entry_notional"]

                    - stock["buy_fee"]

                )

                trade_records.append({

                    "strategy": strategy_name,

                    "signal_date": pos["signal_date"],

                    "entry_date": pos["entry_date"],

                    "exit_date": current_date,

                    "sleeve_id": sleeve_id,

                    "code": code,

                    "entry_price": stock["entry_price"],

                    "exit_price": exit_price,

                    "quantity": stock["quantity"],

                    "entry_notional": stock["entry_notional"],

                    "exit_notional": exit_notional,

                    "buy_fee": stock["buy_fee"],

                    "sell_fee": sell_fee,

                    "net_pnl": net_pnl,

                    "exit_price_source": price_source,

                })

                proceeds_total += exit_notional - sell_fee

            sleeve_cash[sleeve_id] += proceeds_total

            positions[sleeve_id] = None

        # -------------------------------------------------

        # C. 收盘后计算总资产

        # -------------------------------------------------

        total_cash = sum(sleeve_cash.values())

        invested_value = 0.0

        stale_marks_today = 0

        for pos in positions.values():

            if pos is None:

                continue

            for code, stock in pos["stocks"].items():

                mark_price = np.nan

                if code in close_wide.columns:

                    mark_price = close_wide.at[

                        current_date, code

                    ]

                if pd.isna(mark_price) or mark_price <= 0:

                    if code in close_mark.columns:

                        mark_price = close_mark.at[

                            current_date, code

                        ]

                    stale_marks_today += 1

                if pd.notna(mark_price) and mark_price > 0:

                    invested_value += (

                        stock["quantity"] * float(mark_price)

                    )

        total_equity = total_cash + invested_value

        daily_records.append({

            "date": current_date,

            "strategy": strategy_name,

            "cash": total_cash,

            "invested_value": invested_value,

            "equity": total_equity,

            "stale_marks": stale_marks_today,

        })

        # -------------------------------------------------

        # D. 收盘后生成下一交易日订单

        # -------------------------------------------------

        if current_date not in signal_map:

            continue

        idx = date_index[current_date]

        entry_idx = idx + 1

        exit_idx = idx + HOLDING_DAYS

        if exit_idx >= len(trading_dates):

            continue

        entry_date = trading_dates[entry_idx]

        exit_date = trading_dates[exit_idx]

        if entry_date > end_date or exit_date > end_date:

            continue

        # 选一个空闲账户，避免账户轮换冲突

        free_sleeves = [

            i for i in range(HOLDING_DAYS)

            if positions[i] is None

        ]

        already_reserved = {

            item[1] for item in pending.values()

        }

        free_sleeves = [

            i for i in free_sleeves

            if i not in already_reserved

        ]

        if not free_sleeves:

            raise RuntimeError(

                f"{strategy_name}: 没有空闲账户，"

                "请检查信号日期与持仓周期。"

            )

        sleeve_id = free_sleeves[0]

        if entry_date in pending:

            raise RuntimeError("发现重复的待执行开仓订单。")

        pending[entry_date] = (

            current_date,

            sleeve_id,

            signal_map[current_date],

            exit_date,

        )

    # -------------------------------------------------

    # E. 整理每日记录

    # -------------------------------------------------

    daily = pd.DataFrame(daily_records)

    daily = daily[

        (daily["date"] >= start_date)

        & (daily["date"] <= end_date)

    ].copy()

    daily = daily.sort_values("date").reset_index(drop=True)

    if daily.empty:

        raise ValueError(f"{strategy_name}: 每日净值为空。")

    daily["daily_return"] = daily["equity"].pct_change()

    daily.loc[0, "daily_return"] = (

        daily.loc[0, "equity"] / INITIAL_CAPITAL - 1

    )

    daily["drawdown"] = (

        daily["equity"] / daily["equity"].cummax() - 1

    )

    trades = pd.DataFrame(trade_records)

    if not trades.empty:

        trade_pnl = trades["net_pnl"].sum()

    else:

        trade_pnl = 0.0

    final_equity = float(daily["equity"].iloc[-1])

    equity_change = final_equity - INITIAL_CAPITAL

    reconciliation = equity_change - trade_pnl

    all_closed = all(

        pos is None for pos in positions.values()

    )

    if not all_closed:

        raise RuntimeError(

            f"{strategy_name}: 回测结束时仍有持仓，"

            "不能将结果作为完整对照。"

        )

    if pending:

        raise RuntimeError(

            f"{strategy_name}: 回测结束时仍有待执行订单。"

        )

    if abs(reconciliation) > 0.05:

        raise RuntimeError(

            f"{strategy_name}: 资金对账失败，差额 "

            f"{reconciliation:.4f} 元。"

        )

    returns = daily["daily_return"].dropna()

    if len(returns) > 1 and returns.std(ddof=1) > 0:

        sharpe = (

            returns.mean()

            / returns.std(ddof=1)

            * np.sqrt(252)

        )

    else:

        sharpe = np.nan

    summary = {

        "strategy": strategy_name,

        "initial_capital": INITIAL_CAPITAL,

        "final_equity": final_equity,

        "total_return": final_equity / INITIAL_CAPITAL - 1,

        "max_drawdown": daily["drawdown"].min(),

        "annualized_sharpe_rough": sharpe,

        "trade_count": len(trades),

        "trade_net_pnl": trade_pnl,

        "reconciliation_difference": reconciliation,

        "all_positions_closed": all_closed,

        "estimated_exit_count": estimated_exit_count,

        "missing_open_count": missing_open_count,

        "skipped_batches": skipped_batches,

    }

    print(

        f"{strategy_name}: "

        f"期末权益 {final_equity:,.2f} 元，"

        f"收益率 {summary['total_return']:.2%}，"

        f"对账差额 {reconciliation:.6f} 元"

    )

    return daily, trades, summary

# =====================================================

# 6. 运行三种策略

# =====================================================

all_daily = []

all_trades = []

all_summaries = []

for name in STRATEGY_NAMES:

    print(f"\n正在回测：{name}")

    daily, trades, summary = run_backtest(

        signals[name], name

    )

    all_daily.append(daily)

    all_trades.append(trades)

    all_summaries.append(summary)

combined = pd.concat(all_daily, ignore_index=True)

trade_frames = [

    frame for frame in all_trades if not frame.empty

]

if trade_frames:

    combined_trades = pd.concat(

        trade_frames, ignore_index=True

    )

else:

    combined_trades = pd.DataFrame()

summary_df = pd.DataFrame(all_summaries)

combined.to_csv(

    RESULT_FILE, index=False, encoding="utf-8-sig"

)

combined_trades.to_csv(

    TRADE_FILE, index=False, encoding="utf-8-sig"

)

summary_df.to_csv(

    SUMMARY_FILE, index=False, encoding="utf-8-sig"

)

# =====================================================

# 7. 输出结果

# =====================================================

print("\n" + "=" * 75)

print("三种策略统一规则回测结果")

print("=" * 75)

display_cols = [

    "strategy",

    "final_equity",

    "total_return",

    "max_drawdown",

    "annualized_sharpe_rough",

    "trade_count",

    "reconciliation_difference",

]

print(

    summary_df[display_cols].to_string(

        index=False,

        formatters={

            "final_equity": lambda x: f"{x:,.2f}",

            "total_return": lambda x: f"{x:.2%}",

            "max_drawdown": lambda x: f"{x:.2%}",

            "annualized_sharpe_rough": lambda x: f"{x:.3f}",

            "reconciliation_difference": lambda x: f"{x:.6f}",

        },

    )

)

print("\n逐笔交易文件：", TRADE_FILE)

print("每日净值文件：", RESULT_FILE)

print("汇总指标文件：", SUMMARY_FILE)

# =====================================================

# 8. 绘制净值曲线

# =====================================================

plt.figure(figsize=(12, 6))

for name, group in combined.groupby("strategy"):

    group = group.sort_values("date")

    plt.plot(

        group["date"],

        group["equity"] / INITIAL_CAPITAL,

        label=name,

    )

plt.axhline(

    1.0,

    linestyle="--",

    label="Initial Capital",

)

plt.title("Strategy Backtest Comparison")

plt.xlabel("Date")

plt.ylabel("Portfolio Value / Initial Capital")

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(PLOT_FILE, dpi=150)

plt.close()

print("净值曲线文件：", PLOT_FILE)

print("\n回测结束。所有策略均已完成期末空仓检查与资金对账。")