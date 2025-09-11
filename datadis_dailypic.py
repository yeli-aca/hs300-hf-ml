import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================== Path Settings ==================
# （路径设置）
desktop_path = os.path.expanduser("~/Desktop")
stock_file = os.path.join(desktop_path, "沪深300成分数据.xlsx")
index_file = os.path.join(desktop_path, "沪深300指数.xlsx")

# ============== Load Data ==============
# （读取数据）
stock_df = pd.read_excel(stock_file, sheet_name='日度')
index_df = pd.read_excel(index_file, sheet_name='Sheet1')

# 日期格式
stock_df['交易日期'] = pd.to_datetime(stock_df['交易日期'], errors='coerce')
index_df = index_df.rename(columns={index_df.columns[0]: '日期'})
index_df['日期'] = pd.to_datetime(index_df['日期'], errors='coerce')

# 指数字段重命名
index_df.columns = ['日期', 'Open', 'High', 'Low', 'Close', 'Turnover', 'Volume']

# ✅ 数值字段统一转成 float，避免 str/float 错误
num_cols_stock = ['日开盘价','日最高价','日最低价','日收盘价','日个股交易股数','日个股交易金额']
num_cols_index = ['Open','High','Low','Close','Turnover','Volume']

for col in num_cols_stock:
    stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')

for col in num_cols_index:
    index_df[col] = pd.to_numeric(index_df[col], errors='coerce')

# ================== Returns Construction ==================
# （收益率构造）
stock_df = stock_df.sort_values(['证券代码', '交易日期'])
stock_df['Prev_Close'] = stock_df.groupby('证券代码')['日收盘价'].shift(1)
stock_df['Stock_Return'] = np.log(stock_df['日收盘价'] / stock_df['Prev_Close'])

index_df = index_df.sort_values('日期')
index_df['Prev_Close'] = index_df['Close'].shift(1)
index_df['Index_Return'] = np.log(index_df['Close'] / index_df['Prev_Close'])

merged_df = pd.merge(stock_df, index_df[['日期', 'Index_Return']],
                     left_on='交易日期', right_on='日期', how='left')
merged_df['Excess_Return'] = merged_df['Stock_Return'] - merged_df['Index_Return']

# ================== Figure 1: Excess Return Distribution ==================
plt.figure(figsize=(8,5))
sns.histplot(merged_df['Excess_Return'].dropna(), bins=100, kde=True, color="steelblue")
plt.title("Distribution of Excess Returns (Histogram + KDE)")  # 超额收益率分布
plt.xlabel("Excess Return")  # 超额收益率
plt.ylabel("Frequency")      # 频数
plt.tight_layout()
plt.savefig(os.path.join(desktop_path, "Fig1_Excess_Return_Distribution.png"), dpi=300)
plt.show()

# ================== Figure 2: Market Volatility ==================
index_df['Rolling_Vol'] = index_df['Index_Return'].rolling(30).std()
plt.figure(figsize=(10,5))
plt.plot(index_df['日期'], index_df['Rolling_Vol'], label="30-Day Rolling Volatility", color="darkred")
plt.title("HS300 Index Volatility Trend")  # 沪深300指数波动率走势
plt.xlabel("Date")      # 日期
plt.ylabel("Volatility")  # 波动率
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(desktop_path, "Fig2_Market_Volatility.png"), dpi=300)
plt.show()

# ================== Figure 3: Industry Turnover Boxplot ==================
if '申万行业' in stock_df.columns:
    plt.figure(figsize=(10,6))
    sns.boxplot(x='申万行业', y='日个股交易金额', data=stock_df, showfliers=False)
    plt.xticks(rotation=45)
    plt.title("Industry Turnover Distribution (Boxplot)")  # 行业间成交额分布
    plt.ylabel("Turnover")  # 成交额
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, "Fig3_Industry_Turnover_Boxplot.png"), dpi=300)
    plt.show()

# ================== Figure 4: Factor Time Series ==================
# 举例选择几个因子：5日收益率、5日波动率、量比
sample_stock = stock_df['证券代码'].unique()[0]  # 随机选一只股票
sample_df = stock_df[stock_df['证券代码'] == sample_stock].copy()
sample_df['ret_5d'] = np.log(sample_df['日收盘价'] / sample_df['日收盘价'].shift(5))
sample_df['vol_5d'] = sample_df['Stock_Return'].rolling(5).std()
sample_df['volume_ratio'] = sample_df['日个股交易股数'] / sample_df['日个股交易股数'].rolling(5).mean()

plt.figure(figsize=(12,6))
plt.plot(sample_df['交易日期'], sample_df['ret_5d'], label="5-Day Return")
plt.plot(sample_df['交易日期'], sample_df['vol_5d'], label="5-Day Volatility")
plt.plot(sample_df['交易日期'], sample_df['volume_ratio'], label="Volume Ratio")
plt.title(f"Sample Stock {sample_stock} Factor Trends")  # 样本股票因子走势
plt.xlabel("Date")  # 日期
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(desktop_path, "Fig4_Factor_TimeSeries.png"), dpi=300)
plt.show()

# ================== Figure 5: Factor Correlation Heatmap ==================
factor_cols = ['ret_5d','vol_5d','volume_ratio','ma_diff','open_close_diff']
df_corr = merged_df.copy()
df_corr['ret_5d'] = np.log(df_corr['日收盘价'] / df_corr.groupby('证券代码')['日收盘价'].shift(5))
df_corr['vol_5d'] = df_corr.groupby('证券代码')['Stock_Return'].rolling(5).std().reset_index(level=0, drop=True)
df_corr['volume_ratio'] = df_corr.groupby('证券代码')['日个股交易股数'].apply(lambda x: x / x.rolling(5).mean())
df_corr['ma_diff'] = df_corr['日收盘价'] - df_corr.groupby('证券代码')['日收盘价'].rolling(5).mean().reset_index(level=0, drop=True)
df_corr['open_close_diff'] = (df_corr['日收盘价'] - df_corr['日开盘价']) / df_corr['日开盘价']

corr_matrix = df_corr[factor_cols].corr()
sns.clustermap(corr_matrix, cmap="coolwarm", annot=True, figsize=(8,6))
plt.title("Factor Correlation Clustermap")  # 因子相关性聚类热力图
plt.savefig(os.path.join(desktop_path, "Fig5_Factor_Correlation_Clustermap.png"), dpi=300)
plt.show()

# ================== Figure 6: Industry Distribution ==================
if '申万行业' in stock_df.columns:
    industry_dist = stock_df.groupby('申万行业')['证券代码'].nunique()
    plt.figure(figsize=(8,8))
    plt.pie(industry_dist, labels=industry_dist.index, autopct='%1.1f%%', startangle=140)
    plt.title("Industry Distribution of HS300 Sample")  # 沪深300 样本行业分布
    plt.savefig(os.path.join(desktop_path, "Fig6_Industry_Distribution.png"), dpi=300)
    plt.show()
else:
    print("⚠️ Column '申万行业' not found, cannot plot industry distribution.")
