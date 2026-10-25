
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 读取数据
train = pd.read_csv("train_data.csv", parse_dates=["date"])
valid = pd.read_csv("valid_data.csv", parse_dates=["date"])
test = pd.read_csv("test_data.csv", parse_dates=["date"])

features = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
]
target = "target_return_5d"

# 在时间边界留出5个交易日左右的隔离区，
# 避免相邻集合的5日预测目标窗口交叉。
all_dates = sorted(
    pd.to_datetime(
        pd.concat([train["date"], valid["date"], test["date"]]).unique()
    )
)

valid_start = valid["date"].min()
test_start = test["date"].min()

valid_start_pos = all_dates.index(valid_start)
test_start_pos = all_dates.index(test_start)

# 删除验证集开始前5个交易日的训练样本
train_cutoff = all_dates[max(0, valid_start_pos - 5)]
train = train[train["date"] < train_cutoff].copy()

# 删除测试集开始前5个交易日的验证样本
valid_cutoff = all_dates[max(0, test_start_pos - 5)]
valid = valid[valid["date"] < valid_cutoff].copy()

def clean(part):
    part = part.replace([np.inf, -np.inf], np.nan)
    return part.dropna(subset=features + [target])

train = clean(train)
valid = clean(valid)
test = clean(test)

X_train, y_train = train[features], train[target]
X_valid, y_valid = valid[features], valid[target]
X_test, y_test = test[features], test[target]

print("训练样本数：", len(train))
print("验证样本数：", len(valid))
print("测试样本数：", len(test))

# 训练随机森林
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=100,
    random_state=42,
    n_jobs=-1,
)

print("\n开始训练随机森林模型...")
model.fit(X_train, y_train)
print("训练完成！")

def evaluate(name, X, y):
    pred = model.predict(X)

    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))

    # 基准：所有股票都预测为训练集目标收益率均值
    baseline_pred = np.full(len(y), y_train.mean())
    baseline_mae = mean_absolute_error(y, baseline_pred)

    print(f"\n{name}评估")
    print(f"MAE：{mae:.6f}")
    print(f"RMSE：{rmse:.6f}")
    print(f"均值基准 MAE：{baseline_mae:.6f}")

    return pred

# 验证集用于观察模型表现
valid_pred = evaluate("验证集", X_valid, y_valid)

# 测试集只用于最后的样本外评估
test_pred = evaluate("测试集", X_test, y_test)

# 保存模型预测结果
test_result = test[["date", "code", target]].copy()
test_result["predicted_return_5d"] = test_pred
test_result.to_csv(
    "test_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

# 查看模型认为最重要的特征
importance = pd.Series(
    model.feature_importances_,
    index=features,
).sort_values(ascending=False)

print("\n特征重要性：")
print(importance.to_string())

print("\n预测结果已保存：test_predictions.csv")
