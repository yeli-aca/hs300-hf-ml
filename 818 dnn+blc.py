# ✅ 全流程：因子构造 → 标准化 → Lasso / XGBoost / PLS→DNN（IPCA替代）→ SHAP →（可选导出）

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.cross_decomposition import PLSRegression  # 监督式降维，IPCA 的实用替代
from sklearn.decomposition import PCA

import xgboost as xgb
import shap

# ---- (可选) 深度学习 ----
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ============== 基本设置 ==============
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

desktop_path = os.path.expanduser("~/Desktop")
stock_file = os.path.join(desktop_path, "沪深300成分数据.xlsx")
index_file = os.path.join(desktop_path, "沪深300指数.xlsx")

# ============== 读取数据 ==============
stock_df = pd.read_excel(stock_file, sheet_name='日度')
index_df = pd.read_excel(index_file, sheet_name='Sheet1')

# 日期格式
stock_df['交易日期'] = pd.to_datetime(stock_df['交易日期'], errors='coerce')
index_df = index_df.rename(columns={index_df.columns[0]: '日期'})
index_df['日期'] = pd.to_datetime(index_df['日期'], errors='coerce')

# 指数字段重命名 + 数值化
index_df.columns = ['日期', '开盘价', '最高价', '最低价', '收盘价', '成交额', '成交量']
num_cols = ['开盘价', '最高价', '最低价', '收盘价', '成交额', '成交量']
index_df[num_cols] = index_df[num_cols].apply(pd.to_numeric, errors='coerce')

# ============== 收益率构造 ==============
stock_df = stock_df.sort_values(['证券代码', '交易日期'])
stock_df['前日收盘价'] = stock_df.groupby('证券代码')['日收盘价'].shift(1)
stock_df['个股日收益率'] = np.log(stock_df['日收盘价'] / stock_df['前日收盘价'])

index_df = index_df.sort_values('日期')
index_df['前日收盘价'] = index_df['收盘价'].shift(1)
index_df['指数日收益率'] = np.log(index_df['收盘价'] / index_df['前日收盘价'])

# 合并 & 超额收益
merged_df = pd.merge(stock_df, index_df[['日期', '指数日收益率']],
                     left_on='交易日期', right_on='日期', how='left')
merged_df['超额收益率'] = merged_df['个股日收益率'] - merged_df['指数日收益率']

# ============== 因子构造（22个，可再扩展） ==============
df = merged_df.copy().sort_values(['证券代码', '交易日期'])

# 动量类
df['ret_1d'] = np.log(df['日收盘价'] / df.groupby('证券代码')['日收盘价'].shift(1))
df['ret_5d'] = np.log(df['日收盘价'] / df.groupby('证券代码')['日收盘价'].shift(5))
df['ret_10d'] = np.log(df['日收盘价'] / df.groupby('证券代码')['日收盘价'].shift(10))
df['reversal_1d'] = -1 * df.groupby('证券代码')['ret_1d'].shift(1)
df['momentum_ratio'] = df['ret_5d'] / df['ret_10d'].replace(0, np.nan)

# 波动类
df['vol_5d'] = df.groupby('证券代码')['ret_1d'].rolling(5).std().reset_index(level=0, drop=True)
df['vol_10d'] = df.groupby('证券代码')['ret_1d'].rolling(10).std().reset_index(level=0, drop=True)
df['high_low_diff'] = df['日最高价'] - df['日最低价']
df['real_body'] = (df['日收盘价'] - df['日开盘价']).abs()
df['return_vol_ratio'] = df['ret_5d'] / df['vol_5d']

# 成交类
df['volume_ratio'] = df.groupby('证券代码')['日个股交易股数'].apply(lambda x: x / x.rolling(5).mean())
df['volume_std_5d'] = df.groupby('证券代码')['日个股交易股数'].rolling(5).std().reset_index(level=0, drop=True)
df['amount_per_share'] = df['日个股交易金额'] / df['日个股交易股数'].replace(0, np.nan)
df['volume_change_1d'] = df.groupby('证券代码')['日个股交易股数'].apply(lambda x: x / x.shift(1))

# 均线类
df['ma_5'] = df.groupby('证券代码')['日收盘价'].rolling(5).mean().reset_index(level=0, drop=True)
df['ma_10'] = df.groupby('证券代码')['日收盘价'].rolling(10).mean().reset_index(level=0, drop=True)
df['ma_diff'] = df['日收盘价'] - df['ma_5']
df['ma_bias_10'] = (df['日收盘价'] - df['ma_10']) / df['ma_10']

# 结构类
df['open_close_diff'] = (df['日收盘价'] - df['日开盘价']) / df['日开盘价']
df['upper_shadow'] = df['日最高价'] - df[['日开盘价', '日收盘价']].max(axis=1)
df['lower_shadow'] = df[['日开盘价', '日收盘价']].min(axis=1) - df['日最低价']
df['body_range_ratio'] = df['real_body'] / df['high_low_diff'].replace(0, np.nan)

factor_cols = [
    'ret_1d','ret_5d','ret_10d','reversal_1d','momentum_ratio',
    'vol_5d','vol_10d','high_low_diff','real_body','return_vol_ratio',
    'volume_ratio','volume_std_5d','amount_per_share','volume_change_1d',
    'ma_5','ma_10','ma_diff','ma_bias_10',
    'open_close_diff','upper_shadow','lower_shadow','body_range_ratio'
]

# ============== 横截面 Z-score（含零方差保护） ==============
def cross_section_zscore(g):
    mean = g.mean()
    std = g.std(ddof=0)
    if std == 0 or np.isnan(std):
        return (g - mean) * 0
    return (g - mean) / std

for col in factor_cols:
    df[col + '_z'] = df.groupby('交易日期')[col].transform(cross_section_zscore)

factor_cols_z = [c + '_z' for c in factor_cols]

# ============== 建模数据集（统一） ==============
X = df[factor_cols_z]
y = df['超额收益率']
data = pd.concat([X, y], axis=1).dropna()
X_clean = data[factor_cols_z]
y_clean = data['超额收益率']

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=RANDOM_STATE
)

def report_result(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"✅ {name} R²: {r2:.6f}")
    print(f"📉 {name} MSE: {mse:.6e}")
    return r2, mse

# ============== Lasso ==============
lasso = LassoCV(cv=5, random_state=RANDOM_STATE, n_jobs=None)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
report_result("Lasso", y_test, y_pred_lasso)

coef_series = pd.Series(lasso.coef_, index=factor_cols_z)
print("\n📊 Lasso 非零因子（按绝对值排序）:")
print(coef_series[coef_series != 0].sort_values(key=np.abs, ascending=False))

# ============== XGBoost ==============
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
report_result("XGBoost", y_test, y_pred_xgb)

# 特征重要性图
imp = pd.Series(xgb_model.feature_importances_, index=factor_cols_z).sort_values()
plt.figure(figsize=(8,5))
imp.plot(kind='barh')
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# SHAP（对树模型适配良好）
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_clean)
shap.summary_plot(shap_values, X_clean, show=True)

# ============== “IPCA 替代”降维 → DNN ==============
# 方案A（默认）：PLSRegression（监督式因子提炼，近似 IPCA 的“用 y 指导找因子”思想）
USE_PLS = True
N_COMPONENTS = 5  # 可调，建议 3~8

if USE_PLS:
    reducer = PLSRegression(n_components=N_COMPONENTS, scale=False)
    reducer.fit(X_train, y_train)
    Z_train = reducer.transform(X_train)  # 低维新因子
    Z_test  = reducer.transform(X_test)
    reducer_name = f"PLS({N_COMPONENTS})"
else:
    # 方案B：无监督 PCA（如需对比，把 USE_PLS=False）
    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    pca.fit(X_train)
    Z_train = pca.transform(X_train)
    Z_test  = pca.transform(X_test)
    reducer_name = f"PCA({N_COMPONENTS})"
    print("PCA explained variance ratio (sum):", pca.explained_variance_ratio_.sum())

# DNN 使用降维后的输入（更稳、更易学）
dnn = Sequential([
    Dense(64, input_dim=Z_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
dnn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
dnn.fit(Z_train, y_train, epochs=60, batch_size=128, validation_split=0.2, verbose=0)

y_pred_dnn = dnn.predict(Z_test).flatten()
report_result(f"DNN + {reducer_name}", y_test, y_pred_dnn)

# ============== （可选）导出数据 ==============
# output_cols = ['证券代码','交易日期','日收盘价','个股日收益率','指数日收益率','超额收益率'] + factor_cols + factor_cols_z
# output_df = df[output_cols].dropna(subset=['超额收益率'])
# output_df.to_excel(os.path.join(desktop_path, "因子数据_标准化及原始.xlsx"), index=False)
