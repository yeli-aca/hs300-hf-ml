# -*- coding: utf-8 -*-
# 日度：因子构造 → Z-score → Lasso / XGBoost / DNN → SHAP
# + 自动输出基础图表（Fig1/2/5）与表格（Table_D2）

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LassoCV

import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

# ================== 基本设置 ==================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

desktop_path = os.path.expanduser("~/Desktop")
stock_file = os.path.join(desktop_path, "沪深300成分数据.xlsx")  # sheet: 日度
index_file = os.path.join(desktop_path, "沪深300指数.xlsx")     # sheet: Sheet1
out_dir = desktop_path  # 所有输出统一到桌面；如需单独文件夹可改为 os.path.join(desktop_path, "daily_output")

# ================== 读取数据 ==================
stock_df = pd.read_excel(stock_file, sheet_name="日度")
index_df = pd.read_excel(index_file, sheet_name="Sheet1")

# 日期与字段
stock_df["交易日期"] = pd.to_datetime(stock_df["交易日期"], errors="coerce")
index_df = index_df.rename(columns={index_df.columns[0]: "日期"})
index_df["日期"] = pd.to_datetime(index_df["日期"], errors="coerce")
index_df.columns = ["日期", "开盘价", "最高价", "最低价", "收盘价", "成交额", "成交量"]
for c in ["开盘价","最高价","最低价","收盘价","成交额","成交量"]:
    index_df[c] = pd.to_numeric(index_df[c], errors="coerce")

# ================== 收益率构造 ==================
stock_df = stock_df.sort_values(["证券代码","交易日期"])
stock_df["前日收盘价"] = stock_df.groupby("证券代码")["日收盘价"].shift(1)
stock_df["个股日收益率"] = np.log(stock_df["日收盘价"] / stock_df["前日收盘价"])

index_df = index_df.sort_values("日期")
index_df["前日收盘价"] = index_df["收盘价"].shift(1)
index_df["指数日收益率"] = np.log(index_df["收盘价"] / index_df["前日收盘价"])

merged_df = pd.merge(
    stock_df, index_df[["日期","指数日收益率"]],
    left_on="交易日期", right_on="日期", how="left"
)
merged_df["超额收益率"] = merged_df["个股日收益率"] - merged_df["指数日收益率"]

# ================== 基础图表（自动保存） ==================
# Fig1：超额收益率分布
plt.figure(figsize=(8,5))
sns.histplot(merged_df["超额收益率"].dropna(), bins=100, kde=True)
plt.title("Distribution of Excess Returns")
plt.xlabel("Excess Return"); plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig1_Excess_Return_Distribution.png"), dpi=300)
plt.close()

# Fig2：指数滚动波动率（30日）
idx_tmp = index_df.copy().sort_values("日期")
idx_tmp["指数日收益率"] = np.log(idx_tmp["收盘价"] / idx_tmp["收盘价"].shift(1))
idx_tmp["Rolling_Vol_30"] = idx_tmp["指数日收益率"].rolling(30).std()
plt.figure(figsize=(10,5))
plt.plot(idx_tmp["日期"], idx_tmp["Rolling_Vol_30"])
plt.title("HS300 Rolling Volatility (30D)")
plt.xlabel("Date"); plt.ylabel("Volatility")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig2_Market_Volatility.png"), dpi=300)
plt.close()

# ================== 因子构造（22个） ==================
df = merged_df.copy().sort_values(["证券代码","交易日期"])

# 动量类
df["ret_1d"] = np.log(df["日收盘价"] / df.groupby("证券代码")["日收盘价"].shift(1))
df["ret_5d"] = np.log(df["日收盘价"] / df.groupby("证券代码")["日收盘价"].shift(5))
df["ret_10d"] = np.log(df["日收盘价"] / df.groupby("证券代码")["日收盘价"].shift(10))
df["reversal_1d"] = -1 * df.groupby("证券代码")["ret_1d"].shift(1)
df["momentum_ratio"] = df["ret_5d"] / df["ret_10d"].replace(0, np.nan)

# 波动类
df["vol_5d"] = df.groupby("证券代码")["ret_1d"].rolling(5).std().reset_index(level=0, drop=True)
df["vol_10d"] = df.groupby("证券代码")["ret_1d"].rolling(10).std().reset_index(level=0, drop=True)
df["high_low_diff"] = df["日最高价"] - df["日最低价"]
df["real_body"] = (df["日收盘价"] - df["日开盘价"]).abs()
df["return_vol_ratio"] = df["ret_5d"] / df["vol_5d"]

# 成交类
df["volume_ratio"] = df.groupby("证券代码")["日个股交易股数"].apply(lambda x: x / x.rolling(5).mean())
df["volume_std_5d"] = df.groupby("证券代码")["日个股交易股数"].rolling(5).std().reset_index(level=0, drop=True)
df["amount_per_share"] = df["日个股交易金额"] / df["日个股交易股数"].replace(0, np.nan)
df["volume_change_1d"] = df.groupby("证券代码")["日个股交易股数"].apply(lambda x: x / x.shift(1))

# 均线类
df["ma_5"] = df.groupby("证券代码")["日收盘价"].rolling(5).mean().reset_index(level=0, drop=True)
df["ma_10"] = df.groupby("证券代码")["日收盘价"].rolling(10).mean().reset_index(level=0, drop=True)
df["ma_diff"] = df["日收盘价"] - df["ma_5"]
df["ma_bias_10"] = (df["日收盘价"] - df["ma_10"]) / df["ma_10"]

# 形态类
df["open_close_diff"] = (df["日收盘价"] - df["日开盘价"]) / df["日开盘价"]
df["upper_shadow"] = df["日最高价"] - df[["日开盘价","日收盘价"]].max(axis=1)
df["lower_shadow"] = df[["日开盘价","日收盘价"]].min(axis=1) - df["日最低价"]
df["body_range_ratio"] = df["real_body"] / df["high_low_diff"].replace(0, np.nan)

factor_cols = [
    "ret_1d","ret_5d","ret_10d","reversal_1d","momentum_ratio",
    "vol_5d","vol_10d","high_low_diff","real_body","return_vol_ratio",
    "volume_ratio","volume_std_5d","amount_per_share","volume_change_1d",
    "ma_5","ma_10","ma_diff","ma_bias_10",
    "open_close_diff","upper_shadow","lower_shadow","body_range_ratio"
]

# ================== 横截面 Z-score ==================
def cross_section_zscore(g: pd.Series):
    mean = g.mean(); std = g.std(ddof=0)
    if std == 0 or np.isnan(std): return (g - mean) * 0
    return (g - mean) / std

for c in factor_cols:
    df[c + "_z"] = df.groupby("交易日期")[c].transform(cross_section_zscore)

factor_cols_z = [c + "_z" for c in factor_cols]

# Fig5：因子相关性热力图（抽样以防过大）
corr_sample = df[factor_cols_z].dropna()
if len(corr_sample) > 50000:
    corr_sample = corr_sample.sample(50000, random_state=RANDOM_STATE)
corr = corr_sample.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Standardized Factors)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig5_Factor_Correlation_Heatmap.png"), dpi=300)
plt.close()

# ================== 训练/测试划分 ==================
data = pd.concat([df[factor_cols_z], df["超额收益率"]], axis=1).dropna()
X = data[factor_cols_z].astype(float)
y = data["超额收益率"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)

def report_result(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"✅ {name} R²: {r2:.6f}")
    print(f"📉 {name} MSE: {mse:.6e}")
    return r2, mse

# ================== Lasso ==================
lasso = LassoCV(cv=5, random_state=RANDOM_STATE)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
r2_lasso, mse_lasso = report_result("Lasso", y_test, y_pred_lasso)

# 导出【表 D2】：Lasso 非零系数
coef_series = pd.Series(lasso.coef_, index=factor_cols_z)
nz = coef_series[coef_series != 0].sort_values(key=np.abs, ascending=False)
nz.to_csv(os.path.join(out_dir, "Table_D2_Lasso_Nonzero_Coeff.csv"), encoding="utf-8")
print("\n📊 已导出：Table_D2_Lasso_Nonzero_Coeff.csv")

# ================== XGBoost ==================
xgb_model = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb, mse_xgb = report_result("XGBoost", y_test, y_pred_xgb)

# 特征重要性图（保存）
imp = pd.Series(xgb_model.feature_importances_, index=factor_cols_z).sort_values()
plt.figure(figsize=(8,6))
imp.plot(kind="barh")
plt.title("XGBoost Feature Importance (Daily)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig_XGB_Feature_Importance_Daily.png"), dpi=300)
plt.close()

# ================== SHAP（树模型） ==================
try:
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Fig_SHAP_Summary_Daily.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("🖼 已保存：Fig_SHAP_Summary_Daily.png")
except Exception as e:
    print(f"[SHAP] 跳过：{e}")

# ================== DNN（调优版） ==================
dnn = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation="relu", kernel_regularizer=l2(1e-4)),
    BatchNormalization(), Dropout(0.30),
    Dense(64, activation="relu", kernel_regularizer=l2(1e-4)),
    BatchNormalization(), Dropout(0.30),
    Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
    Dropout(0.20),
    Dense(1)
])
dnn.compile(optimizer=Adam(learning_rate=5e-4), loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

hist = dnn.fit(
    X_train, y_train,
    epochs=200, batch_size=128, validation_split=0.2,
    callbacks=[early_stop], verbose=1
)

y_pred_dnn = dnn.predict(X_test).flatten()
r2_dnn, mse_dnn = report_result("Tuned DNN", y_test, y_pred_dnn)

# 学习曲线
plt.figure(figsize=(8,4))
plt.plot(hist.history.get("loss",[]), label="train")
if "val_loss" in hist.history: plt.plot(hist.history["val_loss"], label="val")
plt.title("DNN Learning Curve (MSE)")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Fig_DNN_Learning_Curve_Daily.png"), dpi=300)
plt.close()

# ================== 指标汇总（便于论文表格） ==================
summary = pd.DataFrame({
    "model": ["lasso","xgboost","dnn"],
    "R2": [r2_lasso, r2_xgb, r2_dnn],
    "MSE": [mse_lasso, mse_xgb, mse_dnn]
})
summary.to_csv(os.path.join(out_dir, "Table_Daily_Model_Metrics.csv"), index=False, encoding="utf-8")
print("✅ 已保存：Table_Daily_Model_Metrics.csv")

print("\n[INFO] 所有图片与表格已输出到：", out_dir)
