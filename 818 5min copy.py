# -*- coding: utf-8 -*-
"""
HS300 5-min 高频优化版
- 数据预处理
- 特征构造（微观结构因子）
- 横截面 Z-score
- 滚动窗口训练/测试
- 模型：Lasso / XGBoost / DNN
- 评估：R2, MSE, IC
"""

import os, glob, gc, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ---------------------------
# 配置
# ---------------------------
CONFIG = {
    "STOCK_DIR": "/Users/mengyao/Desktop/5分钟",         # 个股CSV文件夹
    "INDEX_FILE": "/Users/mengyao/Desktop/000300_沪深300_5min.csv",  # 指数CSV
    "OUT_DIR": os.path.expanduser("~/Desktop/hs300_5min_opt"),      # 输出目录
    "ROLL_WINDOW": 20000,   # 滚动训练窗口样本数
    "TEST_SIZE": 2000,      # 每次测试样本数
    "DNN_EPOCHS": 100,
    "DNN_BATCH": 512,
    "SEED": 42
}
os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)

# ---------------------------
# 工具函数
# ---------------------------
def cross_section_zscore(g: pd.Series):
    mean, std = g.mean(), g.std(ddof=0)
    if std == 0 or np.isnan(std): return (g - mean) * 0
    return (g - mean) / std

def ic_score(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]

# ---------------------------
# 读取指数数据
# ---------------------------
def load_index_5min(index_file):
    idx = pd.read_csv(index_file)
    idx.columns = [c.lower() for c in idx.columns]
    idx = idx.rename(columns={"datetime":"datetime","close":"index_close"})
    idx["datetime"] = pd.to_datetime(idx["datetime"], errors="coerce")
    idx["index_close"] = pd.to_numeric(idx["index_close"], errors="coerce")
    idx = idx.dropna().sort_values("datetime")
    idx["ret_mkt"] = np.log(idx["index_close"]/idx["index_close"].shift(1))
    return idx

# ---------------------------
# 特征工程（高频因子）
# ---------------------------
def build_features(panel):
    df = panel.sort_values(["code","datetime"]).copy()
    df["ret_i"] = np.log(df["close"]/df.groupby("code")["close"].shift(1))

    # 高频微观结构因子
    df["hl_range"] = (df["high"]-df["low"])/df["close"]                   # 价差近似
    df["vwap"] = df["amount"]/df["volume"].replace(0,np.nan)
    df["vwap_dev"] = (df["close"]-df["vwap"])/df["vwap"]                  # VWAP偏离
    df["vol_shock"] = df.groupby("code")["volume"].apply(lambda x: x/x.rolling(20).mean())
    df["ret_sign_autocorr"] = df.groupby("code")["ret_i"].apply(
        lambda x: x.rolling(5).apply(lambda s: np.corrcoef(np.sign(s)[:-1], np.sign(s)[1:])[0,1] 
                                      if len(s.dropna())>2 else 0)
    )
    df["volatility_bar"] = df.groupby("code")["ret_i"].rolling(10).std().reset_index(level=0,drop=True)

    factor_cols = ["hl_range","vwap_dev","vol_shock","ret_sign_autocorr","volatility_bar"]
    return df, factor_cols

# ---------------------------
# 构造目标变量
# ---------------------------
def merge_index_and_target(df, idx):
    m = pd.merge(df, idx[["datetime","ret_mkt"]], on="datetime", how="left")
    m["ret_i_t1"] = m.groupby("code")["ret_i"].shift(-1)
    m["ret_mkt_t1"] = m["ret_mkt"].shift(-1)
    m["excess_ret_t1"] = m["ret_i_t1"] - m["ret_mkt_t1"]
    return m

# ---------------------------
# 滚动窗口切分
# ---------------------------
def time_split_rolling(data, roll_window, test_size):
    results = []
    n = len(data)
    for start in range(0, n-roll_window-test_size, test_size):
        train = data.iloc[start:start+roll_window]
        test = data.iloc[start+roll_window:start+roll_window+test_size]
        results.append((train,test))
    return results

# ---------------------------
# 模型
# ---------------------------
def run_lasso(train, test, use_cols):
    from sklearn.linear_model import LassoCV
    X_train,y_train = train[use_cols],train["excess_ret_t1"]
    X_test,y_test = test[use_cols],test["excess_ret_t1"]

    model = LassoCV(cv=3,random_state=CONFIG["SEED"]).fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return {"R2": r2_score(y_test,y_pred),
            "MSE": mean_squared_error(y_test,y_pred),
            "IC": ic_score(y_test,y_pred)}

def run_xgb(train, test, use_cols):
    import xgboost as xgb
    X_train,y_train = train[use_cols],train["excess_ret_t1"]
    X_test,y_test = test[use_cols],test["excess_ret_t1"]

    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=CONFIG["SEED"], n_jobs=-1)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return {"R2": r2_score(y_test,y_pred),
            "MSE": mean_squared_error(y_test,y_pred),
            "IC": ic_score(y_test,y_pred)}

def run_dnn(train, test, use_cols):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    tf.random.set_seed(CONFIG["SEED"])

    X_train,y_train = train[use_cols],train["excess_ret_t1"]
    X_test,y_test = test[use_cols],test["excess_ret_t1"]

    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation="relu"),
        BatchNormalization(), Dropout(0.3),
        Dense(64, activation="relu"), BatchNormalization(), Dropout(0.3),
        Dense(32, activation="relu"), Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=5e-4), loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(X_train,y_train,epochs=CONFIG["DNN_EPOCHS"],
              batch_size=CONFIG["DNN_BATCH"],validation_split=0.2,
              callbacks=[es],verbose=0)
    y_pred = model.predict(X_test,batch_size=CONFIG["DNN_BATCH"]).flatten()
    return {"R2": r2_score(y_test,y_pred),
            "MSE": mean_squared_error(y_test,y_pred),
            "IC": ic_score(y_test,y_pred)}

# ---------------------------
# 主流程
# ---------------------------
def main():
    # 载入股票面板
    files = glob.glob(os.path.join(CONFIG["STOCK_DIR"], "*.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        print(f"\n📂 正在处理文件: {f}")
        print("列名：", df.columns.tolist())

        # 自动识别时间列
        time_col = None
        for c in df.columns:
            if "time" in c.lower() or "date" in c.lower() or "datetime" in c.lower() or "交易" in c:
                time_col = c
                break
        if time_col is None:
            raise ValueError(f"⚠️ 未找到时间列，请检查文件 {f} 的列名: {df.columns.tolist()}")

        # 自动识别代码列
        code_col = None
        for c in df.columns:
            if "code" in c.lower() or "证券" in c or "股票" in c:
                code_col = c
                break
        if code_col is None:
            raise ValueError(f"⚠️ 未找到股票代码列，请检查文件 {f} 的列名: {df.columns.tolist()}")

        # 统一重命名
        rename_map = {
            time_col: "datetime",
            code_col: "code",
            "open": "open", "high": "high", "low": "low", "close": "close",
            "volume": "volume", "amount": "amount"
        }

        # 如果大小写不同，也统一处理
        for col in df.columns:
            lc = col.lower()
            if lc == "open": rename_map[col] = "open"
            if lc == "high": rename_map[col] = "high"
            if lc == "low": rename_map[col] = "low"
            if lc == "close": rename_map[col] = "close"
            if lc in ["volume", "vol"]: rename_map[col] = "volume"
            if lc in ["amount", "turnover"]: rename_map[col] = "amount"

        df = df.rename(columns=rename_map)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["code"] = df["code"].astype(str)

        frames.append(df[["code", "datetime", "open", "high", "low", "close", "volume", "amount"]])

    panel = pd.concat(frames).dropna().sort_values(["code", "datetime"])


        # 指数
    idx = load_index_5min(CONFIG["INDEX_FILE"])

    # 特征工程 + 目标变量
    df, factor_cols = build_features(panel)
    df = merge_index_and_target(df, idx)

    # 横截面Z-score
    for c in factor_cols:
        df[c+"_z"] = df.groupby("datetime")[c].transform(cross_section_zscore)
    use_cols = [c+"_z" for c in factor_cols]

    data = df.dropna(subset=use_cols+["excess_ret_t1"])
    print("最终样本量:", len(data))

    # 滚动窗口训练
    results = []
    for i, (train, test) in enumerate(time_split_rolling(data, CONFIG["ROLL_WINDOW"], CONFIG["TEST_SIZE"])):
        res_lasso = run_lasso(train, test, use_cols)
        res_xgb = run_xgb(train, test, use_cols)
        res_dnn = run_dnn(train, test, use_cols)
        results.append({"fold": i, "lasso": res_lasso, "xgb": res_xgb, "dnn": res_dnn})
        print(f"[Fold {i}]  Lasso R2={res_lasso['R2']:.4f}, XGB R2={res_xgb['R2']:.4f}, DNN R2={res_dnn['R2']:.4f}")

    # 汇总结果保存
    outpath = os.path.join(CONFIG["OUT_DIR"], "rolling_results.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n所有结果已保存到:", outpath)



if __name__=="__main__":
    main()
