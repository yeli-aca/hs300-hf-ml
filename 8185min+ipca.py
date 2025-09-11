# -*- coding: utf-8 -*-
"""
HS300 5-min 全流程脚本（清理修正版）
- 数据预处理（批量 CSV 合并 + 指指数 + 基本面/行业可选）
- 特征构造（OHLCV 技术特征，行业中性化→横截面 Z-score）
- 目标：下一期(5min)个股超额收益 (r_i,t+1 - r_mkt,t+1)
- 模型：LassoCV / XGBoost / DNN(优化版) / IPCA（C1）
- 评估：R2、MSE；图表：分布、波动率、特征重要性、学习曲线、IC、分位组合、成本敏感
"""

import os, gc, glob, json, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error

# C1 的 IPCA 实现
from ipca_core import InstrumentedPCA

# 可选：美化图形
try:
    import seaborn as sns
    SEABORN = True
    sns.set(style="whitegrid")
except Exception:
    SEABORN = False

warnings.filterwarnings("ignore")

# ===================
# 0) 运行参数
# ===================
CONFIG = {
    "STOCK_DIR": "/Users/mengyao/Desktop/5分钟",
    "INDEX_FILE": "/Users/mengyao/Desktop/000300_沪深300_5min.csv",
    "FUND_FILE": "/Users/mengyao/Desktop/季度 (1).xlsx",
    "OUT_DIR": os.path.expanduser("~/Desktop/hs300_5min_output"),

    "ROW_SAMPLE_FRAC": None,
    "TEST_RATIO": 0.2,

    "PLOT": True,
    "DO_SHAP": False,

    "DNN_EPOCHS": 200,
    "DNN_BATCH": 1024,

    "SEED": 42
}

# ===================
# 1) 工具函数
# ===================
def ensure_outdir(path): os.makedirs(path, exist_ok=True)

def parse_datetime(series):
    return pd.to_datetime(series, errors="coerce", utc=False)

def reduce_mem(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_object_dtype(df[col]) and df[col].nunique() / max(len(df[col]),1) < 0.5:
            df[col] = df[col].astype("category")
    return df

def cross_section_zscore(g: pd.Series):
    mean = g.mean(); std = g.std(ddof=0)
    if std == 0 or np.isnan(std): return (g - mean) * 0.0
    return (g - mean) / std

# ===================
# 2) 读数与合并
# ===================
def load_stock_panel(stock_dir: str, sample_frac: float | None = None) -> pd.DataFrame:
    files = glob.glob(os.path.join(stock_dir, "*.csv"))
    if len(files) == 0: raise FileNotFoundError(f"No CSV files found in: {stock_dir}")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            lower = {c.lower(): c for c in df.columns}
            mapping = {
                lower.get("code", "code"): "code",
                lower.get("date", lower.get("datetime", "datetime")): "datetime",
                lower.get("open", "open"): "open",
                lower.get("high", "high"): "high",
                lower.get("low", "low"): "low",
                lower.get("close", "close"): "close",
                lower.get("volume", "volume"): "volume",
                lower.get("turnover", lower.get("amount", "turnover")): "amount",
            }
            df = df.rename(columns=mapping)[list(mapping.values())]
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
    if not os.path.exists(index_file): raise FileNotFoundError(index_file)
    idx = pd.read_csv(index_file)
    lower = {c.lower(): c for c in idx.columns}
    idx = idx.rename(columns={
        lower.get("datetime", lower.get("date", "datetime")): "datetime",
        lower.get("index_close", lower.get("close", "index_close")): "index_close"
    })
    idx["datetime"] = parse_datetime(idx["datetime"])
    idx["index_close"] = pd.to_numeric(idx["index_close"], errors="coerce")
    idx = idx.dropna(subset=["datetime", "index_close"]).sort_values("datetime")
    return reduce_mem(idx)

def load_fundamentals(fund_file: str) -> pd.DataFrame | None:
    if not os.path.exists(fund_file): return None
    fund = pd.read_excel(fund_file) if fund_file.lower().endswith(".xlsx") else pd.read_csv(fund_file)
    lower = {c.lower(): c for c in fund.columns}
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
        if c in fund.columns: fund[c] = pd.to_numeric(fund[c], errors="coerce")
    return fund

# ===================
# 3) 特征与目标
# ===================
def build_features(panel: pd.DataFrame):
    df = panel.sort_values(["code", "datetime"]).copy()
    df["prev_close"] = df.groupby("code")["close"].shift(1)
    df["ret_i"] = np.log(df["close"] / df["prev_close"])
    df["vwap"] = df["amount"] / df["volume"].replace(0, np.nan)

    df["ret_5"]  = df.groupby("code")["close"].apply(lambda x: np.log(x / x.shift(5)))
    df["ret_10"] = df.groupby("code")["close"].apply(lambda x: np.log(x / x.shift(10)))
    df["reversal_1"] = -df.groupby("code")["ret_i"].shift(1)

    df["vol_5"] = df.groupby("code")["ret_i"].rolling(5).std().reset_index(level=0, drop=True)
    df["vol_10"] = df.groupby("code")["ret_i"].rolling(10).std().reset_index(level=0, drop=True)
    df["hl_range"] = (df["high"] - df["low"]) / df["prev_close"]

    df["volume_ratio"] = df.groupby("code")["volume"].apply(lambda x: x / x.rolling(5).mean())
    df["oc_ret"] = (df["close"] - df["open"]) / df["open"]
    df["ma5"] = df.groupby("code")["close"].rolling(5).mean().reset_index(level=0, drop=True)
    df["ma10"] = df.groupby("code")["close"].rolling(10).mean().reset_index(level=0, drop=True)
    df["ma_bias10"] = (df["close"] - df["ma10"]) / df["ma10"]

    df["upper_shadow"] = df["high"] - np.maximum(df["open"], df["close"])
    df["lower_shadow"] = np.minimum(df["open"], df["close"]) - df["low"]
    body = (df["close"] - df["open"]).abs()
    true_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_range_ratio"] = body / true_range

    factor_cols = [
        "ret_5","ret_10","reversal_1",
        "vol_5","vol_10","hl_range",
        "volume_ratio","oc_ret",
        "ma_bias10","upper_shadow","lower_shadow","body_range_ratio"
    ]
    return df, factor_cols

def merge_index_and_target(df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    idx = index_df.copy().sort_values("datetime")
    idx["idx_prev"] = idx["index_close"].shift(1)
    idx["ret_mkt"]  = np.log(idx["index_close"] / idx["idx_prev"])

    m = pd.merge(df, idx[["datetime","ret_mkt"]], on="datetime", how="left")
    m["ret_i_t1"]   = m.groupby("code")["ret_i"].shift(-1)
    m["ret_mkt_t1"] = m["ret_mkt"].shift(-1)
    m["excess_ret_t1"] = m["ret_i_t1"] - m["ret_mkt_t1"]
    return m

def industry_neutralize(df, factor_cols, industry_col="industry"):
    if industry_col not in df.columns: return df, list(factor_cols)
    out_cols = []
    for col in factor_cols:
        neu = df.groupby(["datetime", industry_col])[col].transform(lambda x: x - x.mean())
        newc = col + "_neu"; df[newc] = neu; out_cols.append(newc)
    return df, out_cols

def add_cross_section_zscores(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    for col in factor_cols:
        df[col + "_z"] = df.groupby("datetime")[col].transform(cross_section_zscore)
    return df

# ===================
# 4) 可视化
# ===================
def plot_distribution(y: pd.Series, out_dir: str, title="Excess Return (t+1) Distribution"):
    if not CONFIG["PLOT"]: return
    plt.figure(figsize=(8,5))
    if SEABORN: sns.histplot(y.dropna(), bins=120, kde=True)
    else: plt.hist(y.dropna(), bins=120, alpha=0.85)
    plt.title(title); plt.xlabel("Excess Return (t+1)"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig1_excess_return_dist.png"), dpi=300); plt.close()

def plot_index_vol(index_df: pd.DataFrame, out_dir: str):
    if not CONFIG["PLOT"]: return
    roll = index_df.copy().sort_values("datetime")
    roll["idx_prev"] = roll["index_close"].shift(1)
    roll["ret_mkt"]  = np.log(roll["index_close"] / roll["idx_prev"])
    roll["vol_30"]   = roll["ret_mkt"].rolling(30).std()
    plt.figure(figsize=(10,4)); plt.plot(roll["datetime"], roll["vol_30"])
    plt.title("HS300 5-min Rolling Volatility (30 bars)")
    plt.xlabel("Datetime"); plt.ylabel("Vol")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig2_index_rolling_vol.png"), dpi=300); plt.close()

def plot_xgb_importance(model, feature_names, out_dir: str):
    if not CONFIG["PLOT"]: return
    importances = getattr(model, "feature_importances_", None)
    if importances is None: return
    imp = pd.Series(importances, index=feature_names).sort_values()
    plt.figure(figsize=(8,6)); imp.plot(kind="barh")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig3_xgb_importance.png"), dpi=300); plt.close()

def plot_corr_heatmap(df: pd.DataFrame, feature_names: list[str], out_dir: str):
    if not CONFIG["PLOT"]: return
    samp = df[feature_names].dropna()
    if len(samp) > 50000: samp = samp.sample(50000, random_state=CONFIG["SEED"])
    corr = samp.corr()
    plt.figure(figsize=(8,6))
    if SEABORN: sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    else:
        plt.imshow(corr, cmap="coolwarm"); plt.colorbar()
        plt.xticks(range(len(feature_names)), feature_names, rotation=90)
        plt.yticks(range(len(feature_names)), feature_names)
    plt.title("Correlation Heatmap (sample)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig4_corr_heatmap.png"), dpi=300); plt.close()

def plot_dnn_history(history, out_dir: str):
    if not CONFIG["PLOT"]: return
    plt.figure(figsize=(8,4))
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history: plt.plot(history.history["val_loss"], label="val")
    plt.title("DNN Learning Curve (MSE)"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig5_dnn_learning_curve.png"), dpi=300); plt.close()

# ===================
# 5) 训练与评估
# ===================
def time_folds(df, n_splits=4):
    ts = df["datetime"].sort_values().unique()
    borders = np.linspace(0, len(ts), n_splits+1).astype(int)
    folds = []
    for k in range(1, n_splits+1):
        train_end = ts[borders[k-1]:borders[k]].max()
        test_end  = ts[min(borders[k]+max(1,(borders[1]-borders[0])//4), len(ts)-1)]
        tr = df["datetime"] <= train_end
        te = (df["datetime"] > train_end) & (df["datetime"] <= test_end)
        if te.sum() and tr.sum(): folds.append((df.index[tr], df.index[te]))
    return folds

def wf_eval_lasso(data, use_cols, n_splits=4, seed=42):
    from sklearn.linear_model import LassoCV
    scores = []
    for i, (tr_idx, te_idx) in enumerate(time_folds(data, n_splits=n_splits), 1):
        X_tr, y_tr = data.loc[tr_idx, use_cols], data.loc[tr_idx, "excess_ret_t1"]
        X_te, y_te = data.loc[te_idx, use_cols], data.loc[te_idx, "excess_ret_t1"]
        model = LassoCV(cv=5, random_state=seed).fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        r2 = r2_score(y_te, y_hat); mse = mean_squared_error(y_te, y_hat)
        scores.append((r2, mse)); print(f"[WF-Lasso] fold{i}: R2={r2:.6f} MSE={mse:.6e}")
    if scores:
        r2s, mses = np.array(scores).T
        print(f"[WF-Lasso] mean R2={r2s.mean():.6f} | mean MSE={mses.mean():.6e}")
    return scores

def time_split(df: pd.DataFrame, test_ratio=0.2):
    cutoff = df["datetime"].quantile(1 - test_ratio)
    train = df[df["datetime"] < cutoff].copy()
    test  = df[df["datetime"] >= cutoff].copy()
    return train, test, cutoff

def run_lasso(X_train, y_train, X_test, y_test, out_dir: str):
    from sklearn.linear_model import LassoCV
    model = LassoCV(cv=5, random_state=CONFIG["SEED"]).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred); mse = mean_squared_error(y_test, y_pred)
    coef = pd.Series(model.coef_, index=X_train.columns)
    coef[coef!=0].sort_values(key=np.abs, ascending=False).to_csv(os.path.join(out_dir,"lasso_nonzero_coef.csv"))
    print(f"[Lasso] R2={r2:.6f}  MSE={mse:.6e}")
    return {"r2": r2, "mse": mse, "model": model, "y_pred": y_pred}

def run_xgb(X_train, y_train, X_test, y_test, out_dir: str):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=CONFIG["SEED"],
        n_jobs=-1, reg_lambda=1.0
    ).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred); mse = mean_squared_error(y_test, y_pred)
    print(f"[XGBoost] R2={r2:.6f}  MSE={mse:.6e}")
    plot_xgb_importance(model, list(X_train.columns), out_dir)
    if CONFIG["DO_SHAP"]:
        try:
            import shap
            idx = np.random.RandomState(CONFIG["SEED"]).choice(len(X_train), size=min(20000,len(X_train)), replace=False)
            shap.summary_plot(shap.Explainer(model)(X_train.iloc[idx]), X_train.iloc[idx], show=False)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir,"fig6_shap_summary.png"), dpi=300); plt.close()
        except Exception as e:
            print(f"[SHAP] skip: {e}")
    return {"r2": r2, "mse": mse, "model": model, "y_pred": y_pred}

def run_dnn(X_train, y_train, X_test, y_test, out_dir: str):
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
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=5e-4), loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
    history = model.fit(X_train, y_train, epochs=CONFIG["DNN_EPOCHS"], batch_size=CONFIG["DNN_BATCH"],
                        validation_split=0.2, callbacks=[es, rlrop], verbose=1)
    y_pred = model.predict(X_test, batch_size=CONFIG["DNN_BATCH"]).flatten()
    r2 = r2_score(y_test, y_pred); mse = mean_squared_error(y_test, y_pred)
    print(f"[DNN] R2={r2:.6f}  MSE={mse:.6e}")
    plot_dnn_history(history, out_dir)
    return {"r2": r2, "mse": mse, "model": model, "y_pred": y_pred}

# ---------- IPCA ----------
def build_ipca_data(df, factor_cols_z):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["datetime","code"])
    dates = sorted(df["datetime"].unique()); codes = sorted(df["code"].unique())
    T, N, K = len(dates), len(codes), len(factor_cols_z)
    date_index = {d:i for i,d in enumerate(dates)}; code_index = {c:i for i,c in enumerate(codes)}
    Y = np.full((T,N), np.nan); X = np.full((T,N,K), np.nan)
    for _, row in df.iterrows():
        t = date_index[row["datetime"]]; n = code_index[row["code"]]
        Y[t,n] = row["excess_ret_t1"]
        X[t,n,:] = [row[f] for f in factor_cols_z]
    return Y, X, dates, codes, factor_cols_z

def run_ipca(Y, X, dates, codes, fac_names, out_dir, n_factors=3):
    # 展平成 DataFrame
    T,N,K = Y.shape
    records = []
    for t in range(T):
        for n in range(N):
            if np.isnan(Y[t,n]): continue
            row = {"code": codes[n], "date": dates[t], "y": Y[t,n]}
            for k in range(K): row[fac_names[k]] = X[t,n,k]
            records.append(row)
    df_ipca = pd.DataFrame(records).dropna().reset_index(drop=True)

    df_ipca["id"]   = df_ipca["code"].astype("category").cat.codes
    df_ipca["time"] = df_ipca["date"].astype("category").cat.codes
    df_ipca = df_ipca.set_index(["id","time"])
    y_ipca  = df_ipca["y"].values
    X_ipca  = df_ipca[fac_names].values
    indices = np.array(df_ipca.index.tolist())

    model = InstrumentedPCA(n_factors=n_factors, intercept=True)
    model.fit(X_ipca, y_ipca, indices=indices)
    Gamma, Factors = model.get_factors(label_ind=True)
    Gamma.to_csv(os.path.join(out_dir,"ipca_gamma.csv"))
    Factors.to_csv(os.path.join(out_dir,"ipca_factors.csv"))

    # 预测：优先使用 predict；否则手工重构 y_hat = (X @ Gamma) · f_t
    try:
        y_pred = model.predict(X_ipca, indices=indices)
    except Exception:
        # Gamma: K x r (index=features, columns=factors)；Factors: time x r
        gamma = Gamma.loc[fac_names].values  # K x r
        # 为每一行生成对应的 f_t
        # 取 df_ipca 的 date 列（还在 df_ipca 中，因为我们没删除）
        f_map = {dt: Factors.loc[dt].values for dt in Factors.index}
        f_rows = np.vstack([f_map[dt] for dt in df_ipca.reset_index()["date"]])  # M x r
        y_pred = np.sum((X_ipca @ gamma) * f_rows, axis=1)

    r2 = r2_score(y_ipca, y_pred); mse = mean_squared_error(y_ipca, y_pred)
    print("✅ IPCA 完成 | Gamma:", Gamma.shape, "| Factors:", Factors.shape, f"| R2={r2:.6f} MSE={mse:.6e}")
    return {"r2": r2, "mse": mse, "model": model, "Gamma": Gamma, "Factors": Factors}

# ===================
# 6) IC / 分位 / 成本
# ===================
def compute_ic(df, score_col="score", ret_col="excess_ret_t1"):
    out = (df.dropna(subset=[score_col,ret_col,"datetime","code"])
             .groupby("datetime")
             .apply(lambda g: spearmanr(g[score_col], g[ret_col]).correlation))
    ic_series = out.dropna()
    stats = {"IC_mean": float(ic_series.mean()),
             "IC_std":  float(ic_series.std(ddof=1)),
             "IC_IR":   float(ic_series.mean()/(ic_series.std(ddof=1)+1e-12))}
    return ic_series, stats

def quantile_longshort(df, score_col="score", ret_col="excess_ret_t1", q=5):
    d = df.dropna(subset=[score_col,ret_col,"datetime","code"]).copy()
    d["bucket"] = d.groupby("datetime")[score_col].transform(
        lambda x: pd.qcut(x.rank(method="first"), q, labels=range(1,q+1))
    ).astype(int)
    grp = d.groupby(["datetime","bucket"])[ret_col].mean().unstack("bucket").sort_index()
    grp.columns = [f"Q{c}" for c in grp.columns]
    grp["LS"] = grp[f"Q{q}"] - grp["Q1"]
    cum = (1 + grp).cumprod() - 1
    return grp, cum

def ls_with_cost(grp, est_turnover_bps=0.0002):
    ls_net = grp["LS"] - est_turnover_bps
    cum_net = (1 + ls_net).cumprod() - 1
    return ls_net, cum_net

def plot_ic(ic_series, out_dir, prefix="xgb"):
    plt.figure(figsize=(10,3.5))
    plt.plot(ic_series.index, ic_series.values); plt.axhline(0, ls="--")
    plt.title(f"{prefix.upper()} Spearman IC (5-min bar)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fig_ic_timeseries_{prefix}.png"), dpi=300); plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(ic_series.values, bins=50)
    plt.title(f"{prefix.upper()} IC Distribution"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fig_ic_hist_{prefix}.png"), dpi=300); plt.close()

def plot_quantiles(grp, cum, out_dir, prefix="xgb"):
    plt.figure(figsize=(9,4))
    for c in [c for c in grp.columns if c.startswith("Q") and c!="LS"]:
        plt.plot(grp.index, grp[c], label=c, alpha=0.7)
    plt.title(f"{prefix.upper()} Quantile Portfolio Returns")
    plt.legend(ncol=5, fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fig_quantile_returns_{prefix}.png"), dpi=300); plt.close()

    plt.figure(figsize=(9,4))
    plt.plot(cum.index, cum["LS"])
    plt.title(f"{prefix.upper()} Long-Short (Q{len([c for c in grp.columns if c.startswith('Q')])}-Q1) Cum Return (no cost)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fig_ls_cum_{prefix}.png"), dpi=300); plt.close()

def eval_and_plot_rank(test_df, scores, out_dir, prefix):
    """给任意模型的测试集预测做 IC/分位/成本敏感并出图"""
    d = test_df[["datetime","code","excess_ret_t1"]].copy()
    d["score"] = scores
    ic_series, ic_stats = compute_ic(d, "score", "excess_ret_t1")
    plot_ic(ic_series, out_dir, prefix)
    grp, cum = quantile_longshort(d, "score", "excess_ret_t1", q=5)
    plot_quantiles(grp, cum, out_dir, prefix)
    grp.mean().to_csv(os.path.join(out_dir, f"{prefix}_quantile_mean_returns.csv"))
    ls_net, cum_net = ls_with_cost(grp, est_turnover_bps=0.0002)
    plt.figure(figsize=(9,4)); plt.plot(cum_net.index, cum_net.values)
    plt.title(f"{prefix.upper()} LS Cum Return with Cost (2 bps)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"fig_ls_cum_with_cost_{prefix}.png"), dpi=300); plt.close()
    print(f"[{prefix.upper()}-IC] {ic_stats}")

# ===================
# 7) 模型对比图
# ===================
def plot_model_comparison(summary_json_path, out_dir):
    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    preferred = ["lasso","xgboost","dnn","ipca"]
    models,r2s,mses = [],[],[]
    for m in preferred:
        if m in summary and "R2" in summary[m] and "MSE" in summary[m]:
            models.append(m); r2s.append(float(summary[m]["R2"])); mses.append(float(summary[m]["MSE"]))
    if not models: print("[plot_model_comparison] 无可画指标"); return
    plt.figure(figsize=(6,4)); plt.bar(models, r2s)
    plt.title("Model Comparison: R²"); plt.ylabel("R²")
    for i,v in enumerate(r2s): plt.text(i,v,f"{v:.4f}",ha="center",va="bottom",fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"fig_model_r2_comparison.png"), dpi=300); plt.close()
    plt.figure(figsize=(6,4)); plt.bar(models, mses)
    plt.title("Model Comparison: MSE"); plt.ylabel("MSE")
    for i,v in enumerate(mses): plt.text(i,v,f"{v:.2e}",ha="center",va="bottom",fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"fig_model_mse_comparison.png"), dpi=300); plt.close()

# ===================
# 8) 主流程
# ===================
def main():
    ensure_outdir(CONFIG["OUT_DIR"])

    # 读取
    panel = load_stock_panel(CONFIG["STOCK_DIR"], CONFIG["ROW_SAMPLE_FRAC"])
    idx    = load_index_5min(CONFIG["INDEX_FILE"])
    fund   = load_fundamentals(CONFIG["FUND_FILE"])

    # 特征与目标
    df, factor_cols = build_features(panel); del panel; gc.collect()
    df = merge_index_and_target(df, idx)
    if fund is not None: df = merge_fundamentals(df, fund)

    # 行业中性化 → Z-score（只做一次）
    df, effective_cols = industry_neutralize(df, factor_cols, industry_col="industry")
    df = add_cross_section_zscores(df, effective_cols)
    factor_cols_z = [c + "_z" for c in effective_cols]

    # 训练数据
    use_cols = factor_cols_z
    expected = ["datetime","code","excess_ret_t1"] + use_cols
    data = df[[c for c in expected if c in df.columns]].dropna()
    data = reduce_mem(data)

    # 描述图
    plot_distribution(data["excess_ret_t1"], CONFIG["OUT_DIR"])
    plot_corr_heatmap(data, use_cols, CONFIG["OUT_DIR"])
    plot_index_vol(idx, CONFIG["OUT_DIR"])

    # Walk-forward（可选）
    _ = wf_eval_lasso(data[["datetime","excess_ret_t1"] + use_cols], use_cols, n_splits=4)

    # 时间切分
    train, test, cutoff = time_split(data, CONFIG["TEST_RATIO"])
    X_train, y_train = train[use_cols], train["excess_ret_t1"]
    X_test,  y_test  = test[use_cols],  test["excess_ret_t1"]

    # 训练三模型
    res_lasso = run_lasso(X_train, y_train, X_test, y_test, CONFIG["OUT_DIR"])
    res_xgb   = run_xgb  (X_train, y_train, X_test, y_test, CONFIG["OUT_DIR"])
    res_dnn   = run_dnn  (X_train, y_train, X_test, y_test, CONFIG["OUT_DIR"])

    # IC/分位/成本：可对每个模型都做，这里先演示三份
    eval_and_plot_rank(test, res_lasso["y_pred"], CONFIG["OUT_DIR"], prefix="lasso")
    eval_and_plot_rank(test, res_xgb["y_pred"],   CONFIG["OUT_DIR"], prefix="xgb")
    eval_and_plot_rank(test, res_dnn["y_pred"],   CONFIG["OUT_DIR"], prefix="dnn")

    # IPCA（整段数据 in-sample 评估；如需 OOS，可只用 train 拟合、用 test 重构）
    Y_ipca, X_ipca, dates_ipca, codes_ipca, fac_names_ipca = build_ipca_data(data, use_cols)
    res_ipca = run_ipca(Y=Y_ipca, X=X_ipca, dates=dates_ipca, codes=codes_ipca,
                        fac_names=fac_names_ipca, out_dir=CONFIG["OUT_DIR"], n_factors=3)

    # 指标汇总
    summary = {
        "lasso":  {"R2": float(res_lasso["r2"]), "MSE": float(res_lasso["mse"])},
        "xgboost":{"R2": float(res_xgb["r2"]),   "MSE": float(res_xgb["mse"])},
        "dnn":    {"R2": float(res_dnn["r2"]),   "MSE": float(res_dnn["mse"])},
        "ipca":   {"R2": float(res_ipca["r2"]),  "MSE": float(res_ipca["mse"])},
        "cutoff": str(cutoff),
        "n_train": int(len(train)),
        "n_test":  int(len(test)),
        "n_codes": int(data["code"].nunique())
    }
    with open(os.path.join(CONFIG["OUT_DIR"], "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    plot_model_comparison(os.path.join(CONFIG["OUT_DIR"], "metrics_summary.json"),
                          CONFIG["OUT_DIR"])

if __name__ == "__main__":
    main()
