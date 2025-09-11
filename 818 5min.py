# -*- coding: utf-8 -*-
"""
HS300 5-min 全流程脚本
- 数据预处理（批量 CSV 合并 + 指数字据 + 基本面/行业可选）
- 特征构造（OHLCV 技术特征，横截面 Z-score）
- 目标变量：下一期(5min)个股超额收益 (r_i,t+1 - r_mkt,t+1)
- 模型：LassoCV / XGBoost / DNN(优化版)
- 评估：R2、MSE；图表：分布图、波动率、特征重要性、学习曲线、(可选)SHAP
"""

import os
import gc
import glob
import math
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 可选增强（没有也不影响）
try:
    import seaborn as sns
    SEABORN = True
    sns.set(style="whitegrid")
except Exception:
    SEABORN = False

warnings.filterwarnings("ignore")

# ---------------------------
# 0. 运行参数（已修正格式）
# ---------------------------
CONFIG = {
    # 股票CSV文件夹路径
    "STOCK_DIR": "/Users/mengyao/Desktop/5分钟",

    # 沪深300指数5min行情（请确认已改字段名）
    "INDEX_FILE": "/Users/mengyao/Desktop/000300_沪深300_5min.csv",

    # 基本面（请确认已改字段名）
    "FUND_FILE": "/Users/mengyao/Desktop/季度 (1).xlsx",

    # 输出目录
    "OUT_DIR": os.path.expanduser("~/Desktop/hs300_5min_output"),

    # 抽样比例（None为不抽样）
    "ROW_SAMPLE_FRAC": None,

    # 测试集比例
    "TEST_RATIO": 0.2,

    # 是否画图
    "PLOT": True,

    # 是否做SHAP解释（很慢）
    "DO_SHAP": False,

    # DNN参数
    "DNN_EPOCHS": 200,
    "DNN_BATCH": 1024,

    # 随机种子
    "SEED": 42
}


# ---------------------------
# 剩余代码（不变）
# ---------------------------



# ---------------------------
# 1. 工具函数
# ---------------------------

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def parse_datetime(series):
    """把多种常见字段统一成 pandas datetime (秒级)。"""
    # 自动识别/兼容 2022-01-01 09:35:00 / 2022/01/01 09:35 等
    return pd.to_datetime(series, errors="coerce", utc=False)

def reduce_mem(df):
    """简单降内存：整数/浮点降位宽；字符串转category。"""
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_object_dtype(df[col]) and df[col].nunique() / max(len(df[col]),1) < 0.5:
            df[col] = df[col].astype("category")
    return df

def cross_section_zscore(g: pd.Series):
    """横截面 Z-score（同一时间截面内对所有股票）"""
    mean = g.mean()
    std = g.std(ddof=0)
    if std == 0 or np.isnan(std):
        return (g - mean) * 0.0
    return (g - mean) / std

# ---------------------------
# 2. 数据读取与合并
# ---------------------------

def load_stock_panel(stock_dir: str,
                     sample_frac: float | None = None) -> pd.DataFrame:
    """
    批量读取 5min 个股 CSV，统一字段：
      code, datetime, open, high, low, close, volume, amount
    """
    files = glob.glob(os.path.join(stock_dir, "*.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in: {stock_dir}")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # 字段名兼容：date/datetime, turnover/amount
            cols = {c.lower(): c for c in df.columns}
            # 必要列映射
            mapping = {
                cols.get("code", "code"): "code",
                cols.get("date", cols.get("datetime", "datetime")): "datetime",
                cols.get("open", "open"): "open",
                cols.get("high", "high"): "high",
                cols.get("low", "low"): "low",
                cols.get("close", "close"): "close",
                cols.get("volume", "volume"): "volume",
                cols.get("turnover", cols.get("amount", "turnover")): "amount",
            }
            df = df.rename(columns=mapping)[list(mapping.values())]
            # to numeric
            for c in ["open", "high", "low", "close", "volume", "amount"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["datetime"] = parse_datetime(df["datetime"])
            df["code"] = df["code"].astype(str)
            if sample_frac is not None:
                df = df.sample(frac=sample_frac, random_state=CONFIG["SEED"])
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skip file {f}: {e}")

    panel = pd.concat(frames, axis=0, ignore_index=True)
    panel = panel.dropna(subset=["datetime", "code"]).sort_values(["code", "datetime"])
    panel = reduce_mem(panel)
    return panel


def load_index_5min(index_file: str) -> pd.DataFrame:
    """读取指数 5min 数据，统一字段：datetime, index_close"""
    if not os.path.exists(index_file):
        raise FileNotFoundError(index_file)
    idx = pd.read_csv(index_file)
    # 字段名兼容
    lower = {c.lower(): c for c in idx.columns}
    idx = idx.rename(columns={
        lower.get("datetime", lower.get("date", "datetime")): "datetime",
        lower.get("index_close", lower.get("close", "index_close")): "index_close"
    })
    idx["datetime"] = parse_datetime(idx["datetime"])
    idx["index_close"] = pd.to_numeric(idx["index_close"], errors="coerce")
    idx = idx.dropna(subset=["datetime", "index_close"]).sort_values("datetime")
    idx = reduce_mem(idx)
    return idx


def load_fundamentals(fund_file: str) -> pd.DataFrame | None:
    """读取行业&基本面（可选）。支持 xlsx/csv。"""
    if not os.path.exists(fund_file):
        return None
    if fund_file.lower().endswith(".xlsx"):
        fund = pd.read_excel(fund_file)
    else:
        fund = pd.read_csv(fund_file)

    lower = {c.lower(): c for c in fund.columns}
    # 至少包括 code, date(or period), industry，可选财务字段
    fund = fund.rename(columns={
        lower.get("code", "code"): "code",
        lower.get("date", lower.get("period", "date")): "date",
        lower.get("industry", "industry"): "industry",
        lower.get("market_cap", "market_cap"): "market_cap",
        lower.get("pe_ttm", "pe_ttm"): "pe_ttm",
        lower.get("pb", "pb"): "pb",
        lower.get("roe", "roe"): "roe",
        lower.get("total_asset", "total_asset"): "total_asset",
    })
    fund["code"] = fund["code"].astype(str)
    fund["date"] = pd.to_datetime(fund["date"], errors="coerce")
    for c in ["market_cap", "pe_ttm", "pb", "roe", "total_asset"]:
        if c in fund.columns:
            fund[c] = pd.to_numeric(fund[c], errors="coerce")
    return fund


# ---------------------------
# 3. 特征工程与目标变量
# ---------------------------

def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    在个股 5min 面板上基于 OHLCV 构造技术特征（滚动窗口注意以 code 分组）
    返回：加入特征列和个股收益 r_i,t
    """
    df = panel.sort_values(["code", "datetime"]).copy()

    # 个股 5min 对数收益
    df["prev_close"] = df.groupby("code")["close"].shift(1)
    df["ret_i"] = np.log(df["close"] / df["prev_close"])

    # vwap 近似（金额/量）
    df["vwap"] = df["amount"] / df["volume"].replace(0, np.nan)

    # 窗口（可按需扩展）
    def g_rolling(s, win, func):
        return s.rolling(win, min_periods=win).apply(func, raw=False)

    # 动量类
    df["ret_5"] = df.groupby("code")["close"].apply(lambda x: np.log(x / x.shift(5)))
    df["ret_10"] = df.groupby("code")["close"].apply(lambda x: np.log(x / x.shift(10)))
    df["reversal_1"] = -df.groupby("code")["ret_i"].shift(1)

    # 波动/区间类
    df["vol_5"] = df.groupby("code")["ret_i"].rolling(5).std().reset_index(level=0, drop=True)
    df["vol_10"] = df.groupby("code")["ret_i"].rolling(10).std().reset_index(level=0, drop=True)
    df["hl_range"] = (df["high"] - df["low"]) / df["prev_close"]

    # 量价
    df["volume_ratio"] = df.groupby("code")["volume"].apply(lambda x: x / x.rolling(5).mean())
    df["oc_ret"] = (df["close"] - df["open"]) / df["open"]
    df["ma5"] = df.groupby("code")["close"].rolling(5).mean().reset_index(level=0, drop=True)
    df["ma10"] = df.groupby("code")["close"].rolling(10).mean().reset_index(level=0, drop=True)
    df["ma_bias10"] = (df["close"] - df["ma10"]) / df["ma10"]

    # K 线形态
    df["upper_shadow"] = df["high"] - np.maximum(df["open"], df["close"])
    df["lower_shadow"] = np.minimum(df["open"], df["close"]) - df["low"]
    body = (df["close"] - df["open"]).abs()
    true_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_range_ratio"] = body / true_range

    # 选取输出的因子列
    factor_cols = [
        "ret_5", "ret_10", "reversal_1",
        "vol_5", "vol_10", "hl_range",
        "volume_ratio", "oc_ret",
        "ma_bias10", "upper_shadow", "lower_shadow", "body_range_ratio"
    ]
    return df, factor_cols


def merge_index_and_target(df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并指数收益并计算目标变量（下一期 5min 超额收益）
    y_{t+1} = (r_i,t+1 - r_mkt,t+1)
    """
    idx = index_df.copy()
    idx = idx.sort_values("datetime")
    idx["idx_prev"] = idx["index_close"].shift(1)
    idx["ret_mkt"] = np.log(idx["index_close"] / idx["idx_prev"])

    m = pd.merge(df, idx[["datetime", "ret_mkt"]], on="datetime", how="left")
    # 目标变量用 t+1：避免未来信息泄露
    m["ret_i_t1"] = m.groupby("code")["ret_i"].shift(-1)
    m["ret_mkt_t1"] = m["ret_mkt"].shift(-1)
    m["excess_ret_t1"] = m["ret_i_t1"] - m["ret_mkt_t1"]
    return m


def add_cross_section_zscores(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """对每个时间截面的所有股票做横截面 Z-score 标准化"""
    for col in factor_cols:
        df[col + "_z"] = df.groupby("datetime")[col].transform(cross_section_zscore)
    return df


def merge_fundamentals(df: pd.DataFrame, fund: pd.DataFrame | None) -> pd.DataFrame:
    """把行业与基本面合并到 5min 面板（按最近可得对齐：asof merge）。"""
    if fund is None:
        return df

    fund = fund.sort_values(["code", "date"])
    # asof 需要键合并（每个 code 独立）
    out = []
    for code, g in df.groupby("code", sort=False):
        gg = g.sort_values("datetime")
        ff = fund[fund["code"] == code].sort_values("date")
        if len(ff) == 0:
            out.append(gg)
            continue
        # 近似“用当时点之前最近一期的财务/行业”
        merged = pd.merge_asof(
            gg, ff, left_on="datetime", right_on="date",
            direction="backward", allow_exact_matches=True
        )
        out.append(merged.drop(columns=["date"], errors="ignore"))
    res = pd.concat(out, axis=0, ignore_index=True)
    return res

# ---------------------------
# 4. 可视化（保存 PNG）
# ---------------------------

def plot_distribution(y: pd.Series, out_dir: str, title="Excess Return (t+1) Distribution"):
    if not CONFIG["PLOT"]:
        return
    plt.figure(figsize=(8,5))
    if SEABORN:
        sns.histplot(y.dropna(), bins=120, kde=True)
    else:
        plt.hist(y.dropna(), bins=120, alpha=0.85)
    plt.title(title)
    plt.xlabel("Excess Return (t+1)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_excess_return_dist.png"), dpi=300)
    plt.close()

def plot_index_vol(index_df: pd.DataFrame, out_dir: str):
    if not CONFIG["PLOT"]:
        return
    roll = index_df.copy().sort_values("datetime")
    roll["idx_prev"] = roll["index_close"].shift(1)
    roll["ret_mkt"] = np.log(roll["index_close"] / roll["idx_prev"])
    roll["vol_30"] = roll["ret_mkt"].rolling(30).std()

    plt.figure(figsize=(10,4))
    plt.plot(roll["datetime"], roll["vol_30"])
    plt.title("HS300 5-min Rolling Volatility (30 bars)")
    plt.xlabel("Datetime"); plt.ylabel("Vol")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_index_rolling_vol.png"), dpi=300)
    plt.close()

def plot_xgb_importance(model, feature_names, out_dir: str):
    if not CONFIG["PLOT"]:
        return
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    imp = pd.Series(importances, index=feature_names).sort_values()
    plt.figure(figsize=(8,6))
    imp.plot(kind="barh")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_xgb_importance.png"), dpi=300)
    plt.close()

def plot_corr_heatmap(df: pd.DataFrame, feature_names: list[str], out_dir: str):
    if not CONFIG["PLOT"]:
        return
    # 为了内存和速度，仅抽样画热力图
    samp = df[feature_names].dropna().sample(min(50000, df[feature_names].dropna().shape[0]),
                                            random_state=CONFIG["SEED"])
    corr = samp.corr()
    plt.figure(figsize=(8,6))
    if SEABORN:
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    else:
        plt.imshow(corr, cmap="coolwarm"); plt.colorbar()
        plt.xticks(range(len(feature_names)), feature_names, rotation=90)
        plt.yticks(range(len(feature_names)), feature_names)
    plt.title("Correlation Heatmap (sample)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_corr_heatmap.png"), dpi=300)
    plt.close()

def plot_dnn_history(history, out_dir: str):
    if not CONFIG["PLOT"]:
        return
    plt.figure(figsize=(8,4))
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.title("DNN Learning Curve (MSE)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_dnn_learning_curve.png"), dpi=300)
    plt.close()

# ---------------------------
# 5. 机器学习建模
# ---------------------------

def time_split(df: pd.DataFrame, test_ratio=0.2):
    """按时间切分训练/测试，避免泄露"""
    cutoff = df["datetime"].quantile(1 - test_ratio)
    train = df[df["datetime"] < cutoff].copy()
    test = df[df["datetime"] >= cutoff].copy()
    return train, test, cutoff

def run_lasso(X_train, y_train, X_test, y_test, out_dir: str):
    from sklearn.linear_model import LassoCV
    from sklearn.metrics import r2_score, mean_squared_error

    model = LassoCV(cv=5, random_state=CONFIG["SEED"], n_jobs=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    coef = pd.Series(model.coef_, index=X_train.columns)
    coef_nonzero = coef[coef != 0].sort_values(key=np.abs, ascending=False)
    coef_nonzero.to_csv(os.path.join(out_dir, "lasso_nonzero_coef.csv"))
    print(f"[Lasso] R2={r2:.6f}  MSE={mse:.6e}")
    return {"r2": r2, "mse": mse, "model": model}

def run_xgb(X_train, y_train, X_test, y_test, out_dir: str):
    from sklearn.metrics import r2_score, mean_squared_error
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=CONFIG["SEED"],
        n_jobs=-1,
        reg_lambda=1.0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"[XGBoost] R2={r2:.6f}  MSE={mse:.6e}")

    plot_xgb_importance(model, list(X_train.columns), out_dir)

    # 可选：SHAP
    if CONFIG["DO_SHAP"]:
        try:
            import shap
            explainer = shap.Explainer(model)
            # 抽样 2 万做解释，避免过慢
            idx = np.random.RandomState(CONFIG["SEED"]).choice(
                len(X_train), size=min(20000, len(X_train)), replace=False
            )
            shap_values = explainer(X_train.iloc[idx])
            shap.summary_plot(shap_values, X_train.iloc[idx], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig6_shap_summary.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"[SHAP] skip: {e}")

    return {"r2": r2, "mse": mse, "model": model}

def run_dnn(X_train, y_train, X_test, y_test, out_dir: str):
    from sklearn.metrics import r2_score, mean_squared_error
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    tf.random.set_seed(CONFIG["SEED"])

    model = Sequential([
        Dense(256, input_dim=X_train.shape[1], activation="relu", kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Dropout(0.3),
        Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Dropout(0.3),
        Dense(64, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(1)  # 回归
    ])
    model.compile(optimizer=Adam(learning_rate=5e-4), loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        epochs=CONFIG["DNN_EPOCHS"],
        batch_size=CONFIG["DNN_BATCH"],
        validation_split=0.2,
        callbacks=[es, rlrop],
        verbose=1
    )
    y_pred = model.predict(X_test, batch_size=CONFIG["DNN_BATCH"]).flatten()
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"[DNN] R2={r2:.6f}  MSE={mse:.6e}")

    # 学习曲线
    plot_dnn_history(history, out_dir)
    return {"r2": r2, "mse": mse, "model": model}

# ---------------------------
# 6. 主流程
# ---------------------------

def main():
    ensure_outdir(CONFIG["OUT_DIR"])
    print(">> Loading stock 5-min panel ...")
    panel = load_stock_panel(CONFIG["STOCK_DIR"], CONFIG["ROW_SAMPLE_FRAC"])

    if CONFIG["ROW_SAMPLE_FRAC"] is not None:
        print(f"⚠️ 当前启用了抽样，使用比例 = {CONFIG['ROW_SAMPLE_FRAC']}")

    print("🟢 原始面板数据检查:")
    print("总行数：", len(panel))
    print("股票数量：", panel["code"].nunique())
    print("时间范围：", panel["datetime"].min(), "~", panel["datetime"].max())
    print("平均每只股票数据量：", len(panel) // panel["code"].nunique())

    print(f"   stocks rows = {len(panel):,}, stocks = {panel['code'].nunique()} "
          f"range = [{panel['datetime'].min()} ~ {panel['datetime'].max()}]")

    print(">> Loading HS300 index 5-min ...")
    idx = load_index_5min(CONFIG["INDEX_FILE"])
    print(f"   index rows  = {len(idx):,} range = [{idx['datetime'].min()} ~ {idx['datetime'].max()}]")

    # 可选基本面/行业
    fund = load_fundamentals(CONFIG["FUND_FILE"])
    if fund is None:
        print(">> Fundamentals not provided (industry/financials skipped).")
    else:
        print(f">> Fundamentals loaded: rows={len(fund):,}, codes={fund['code'].nunique()}")

    # 特征工程
    print(">> Building features ...")
    df, factor_cols = build_features(panel)
    del panel; gc.collect()

    # 合并指数并构造目标变量
    print(">> Merging index & building target (t+1 excess return) ...")
    df = merge_index_and_target(df, idx)

    # 合并基本面/行业
    if fund is not None:
        print(">> Merging fundamentals (asof) ...")
        df = merge_fundamentals(df, fund)

    # 横截面 Z-score（只对技术因子做）
    print(">> Cross-section z-scoring ...")
    df = add_cross_section_zscores(df, factor_cols)
    factor_cols_z = [c + "_z" for c in factor_cols]
    print("🟡 特征工程完成后数据量：", len(df))



       # 选择训练数据列
    use_cols = factor_cols_z  # 如需把基本面加入，可 append 基本面字段名（注意做标准化或缩放）

    # ✅ 调试：先打印列名确认 code 存在
    print("✔ 当前 df 列名：", df.columns.tolist())

    # ✅ 防止列缺失崩溃（更稳健）
    expected_cols = ["datetime", "code", "excess_ret_t1"] + use_cols
    data = df[[col for col in expected_cols if col in df.columns]].dropna()
    print("🔵 实际用于建模的数据量：", len(data))
    print("使用率：{:.2%}".format(len(data) / len(df)))


    # 降内存
    data = reduce_mem(data)

    # 保存一些描述图
    plot_distribution(data["excess_ret_t1"], CONFIG["OUT_DIR"])
    plot_corr_heatmap(data, use_cols, CONFIG["OUT_DIR"])
    plot_index_vol(idx, CONFIG["OUT_DIR"])

    # 时间切分
    train, test, cutoff = time_split(data, CONFIG["TEST_RATIO"])
    print(f">> Train before {cutoff}, rows={len(train):,} | Test after {cutoff}, rows={len(test):,}")

    X_train = train[use_cols]; y_train = train["excess_ret_t1"]
    X_test = test[use_cols];   y_test  = test["excess_ret_t1"]

    print("🟣 模型输入维度检查：")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # ---- Lasso
    res_lasso = run_lasso(X_train, y_train, X_test, y_test, CONFIG["OUT_DIR"])

    # ---- XGBoost
    res_xgb = run_xgb(X_train, y_train, X_test, y_test, CONFIG["OUT_DIR"])

    # ---- DNN
    res_dnn = run_dnn(X_train, y_train, X_test, y_test, CONFIG["OUT_DIR"])

    # 结果汇总
    summary = {
        "lasso": {"R2": float(res_lasso["r2"]), "MSE": float(res_lasso["mse"])},
        "xgboost": {"R2": float(res_xgb["r2"]), "MSE": float(res_xgb["mse"])},
        "dnn": {"R2": float(res_dnn["r2"]), "MSE": float(res_dnn["mse"])},
        "cutoff": str(cutoff),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_codes": int(data["code"].nunique()) if "code" in data.columns else 0
    }
    with open(os.path.join(CONFIG["OUT_DIR"], "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nAll outputs saved to: {CONFIG['OUT_DIR']}")
    # 放到 main() 最后执行
    plot_model_comparison(
        summary_json_path=os.path.join(CONFIG["OUT_DIR"], "metrics_summary.json"),
        out_dir=CONFIG["OUT_DIR"]
    )


# ---------------------------
# 7. 附加图表：模型性能对比
# ---------------------------

def plot_model_comparison(summary_json_path, out_dir):
    import matplotlib.pyplot as plt
    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    models = ["lasso", "xgboost", "dnn"]
    r2_scores = [summary[m]["R2"] for m in models]
    mse_scores = [summary[m]["MSE"] for m in models]

    # R² 对比图
    plt.figure(figsize=(6, 4))
    plt.bar(models, r2_scores, color='steelblue')
    plt.title("Model Comparison: R²")
    plt.ylabel("R²")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_model_r2_comparison.png"), dpi=300)
    plt.close()

    # MSE 对比图
    plt.figure(figsize=(6, 4))
    plt.bar(models, mse_scores, color='indianred')
    plt.title("Model Comparison: MSE")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_model_mse_comparison.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
