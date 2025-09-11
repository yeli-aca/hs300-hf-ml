# utils.py
# 基础工具：IO、特征工程、图表与评估
import os
import glob
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

CONFIG = {
    "STOCK_DIR": "/Users/mengyao/Desktop/5分钟",
    "INDEX_FILE": "/Users/mengyao/Desktop/000300_沪深300_5min.csv",
    "FUND_FILE": "/Users/mengyao/Desktop/季度 (1).xlsx",
    "OUT_DIR": os.path.expanduser("~/Desktop/hs300_5min_output"),
    "ROW_SAMPLE_FRAC": None,   # e.g. 0.2 做抽样加速
    "TEST_RATIO": 0.2,
    "PLOT": True,
    "DO_SHAP": False,
    "DNN_EPOCHS": 200,
    "DNN_BATCH": 1024,
    "SEED": 42,
}

# ---------------- 工具函数 ----------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def parse_datetime(series):
    return pd.to_datetime(series, errors="coerce", utc=False)

def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_object_dtype(df[col]) and df[col].nunique() / max(len(df[col]), 1) < 0.5:
            df[col] = df[col].astype("category")
    return df

def cross_section_zscore(g: pd.Series):
    mean = g.mean()
    std = g.std(ddof=0)
    if std == 0 or np.isnan(std):
        return (g - mean) * 0.0
    return (g - mean) / std

def safe_cols(df, cols):
    """确保列存在时才返回"""
    return [c for c in cols if c in df.columns]

# ---------------- I/O ----------------
def load_stock_panel(stock_dir: str, sample_frac=None) -> pd.DataFrame:
    files = glob.glob(os.path.join(stock_dir, "*.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in: {stock_dir}")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            cols = {c.lower(): c for c in df.columns}
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
            for c in ["open", "high", "low", "close", "volume", "amount"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["datetime"] = parse_datetime(df["datetime"])
            df["code"] = df["code"].astype(str)
            if sample_frac is not None:
                df = df.sample(frac=sample_frac, random_state=CONFIG["SEED"])
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skip file {f}: {e}")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["datetime", "code"]).sort_values(["code", "datetime"])
    return reduce_mem(panel)

def load_index_5min(index_file: str) -> pd.DataFrame:
    if not os.path.exists(index_file):
        raise FileNotFoundError(index_file)
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
    if not os.path.exists(fund_file):
        return None
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
        if c in fund.columns:
            fund[c] = pd.to_numeric(fund[c], errors="coerce")
    return fund

# ------------- 特征工程 -------------
def build_features(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = panel.sort_values(["code", "datetime"]).copy()

    df["prev_close"] = df.groupby("code")["close"].shift(1)
    df["ret_i"] = np.log(df["close"] / df["prev_close"])

    df["vwap"] = df["amount"] / df["volume"].replace(0, np.nan)

    df["ret_5"]  = df.groupby("code")["close"].apply(lambda x: np.log(x / x.shift(5)))
    df["ret_10"] = df.groupby("code")["close"].apply(lambda x: np.log(x / x.shift(10)))

    df["reversal_1"] = -df.groupby("code")["ret_i"].shift(1)

    df["vol_5"]  = df.groupby("code")["ret_i"].rolling(5).std().reset_index(level=0, drop=True)
    df["vol_10"] = df.groupby("code")["ret_i"].rolling(10).std().reset_index(level=0, drop=True)

    df["hl_range"] = (df["high"] - df["low"]) / df["prev_close"]
    df["volume_ratio"] = df.groupby("code")["volume"].apply(lambda x: x / x.rolling(5).mean())

    df["oc_ret"] = (df["close"] - df["open"]) / df["open"]

    df["ma10"] = df.groupby("code")["close"].rolling(10).mean().reset_index(level=0, drop=True)
    df["ma_bias10"] = (df["close"] - df["ma10"]) / df["ma10"]

    df["upper_shadow"] = df["high"] - np.maximum(df["open"], df["close"])
    df["lower_shadow"] = np.minimum(df["open"], df["close"]) - df["low"]

    body = (df["close"] - df["open"]).abs()
    true_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_range_ratio"] = body / true_range

    factor_cols = [
        "ret_5", "ret_10", "reversal_1", "vol_5", "vol_10", "hl_range",
        "volume_ratio", "oc_ret", "ma_bias10", "upper_shadow", "lower_shadow", "body_range_ratio"
    ]
    return df, factor_cols

def merge_index_and_target(df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    # 统一 code 列名
    if "code" not in df.columns:
        if "code_x" in df.columns:   df = df.rename(columns={"code_x": "code"})
        elif "code_y" in df.columns: df = df.rename(columns={"code_y": "code"})
        elif "stock_code" in df.columns: df = df.rename(columns={"stock_code": "code"})
        elif "代码" in df.columns:   df = df.rename(columns={"代码": "code"})

    idx = index_df.sort_values("datetime").copy()
    idx["idx_prev"]   = idx["index_close"].shift(1)
    idx["ret_mkt"]    = np.log(idx["index_close"] / idx["idx_prev"])
    idx["ret_mkt_t1"] = idx["ret_mkt"].shift(-1)  # 关键：把市场收益对齐到 t+1

    m = pd.merge(df, idx[["datetime", "ret_mkt", "ret_mkt_t1"]], on="datetime", how="left")

    if "ret_i" not in m.columns:
        raise ValueError("缺少个股收益率列 ret_i，无法计算超额收益")

    m["ret_i_t1"] = m.groupby("code")["ret_i"].shift(-1)
    m["excess_ret_t1"] = m["ret_i_t1"] - m["ret_mkt_t1"]

    assert "code" in m.columns, "[merge_index_and_target] 缺少 code 列"
    return m

def merge_fundamentals(df: pd.DataFrame, fund: pd.DataFrame | None) -> pd.DataFrame:
    """按代码进行 as-of 合并基本面数据（季度/半年度），在每个 datetime 使用最近一期已披露值。"""
    if fund is None:
        return df

    # 标准化 code
    if "code" not in df.columns:
        if "code_x" in df.columns:
            df = df.rename(columns={"code_x": "code"})
        elif "code_y" in df.columns:
            df = df.rename(columns={"code_y": "code"})
        elif "stock_code" in df.columns:
            df = df.rename(columns={"stock_code": "code"})
        elif "代码" in df.columns:
            df = df.rename(columns={"代码": "code"})

    if "code" not in fund.columns:
        if "stock_code" in fund.columns:
            fund = fund.rename(columns={"stock_code": "code"})
        elif "代码" in fund.columns:
            fund = fund.rename(columns={"代码": "code"})

    fund = fund.sort_values(["code", "date"])

    out = []
    for code, g in df.groupby("code", sort=False):
        gg = g.sort_values("datetime")
        ff = fund[fund["code"] == code].sort_values("date")
        if ff.empty:
            out.append(gg)
            continue

        merged = pd.merge_asof(
            gg, ff, left_on="datetime", right_on="date",
            direction="backward", allow_exact_matches=True
        )
        merged = merged.drop(columns=["date"], errors="ignore")

        if "code_x" in merged.columns:
            merged = merged.rename(columns={"code_x": "code"})
        if "code_y" in merged.columns:
            merged = merged.drop(columns=["code_y"], errors="ignore")

        out.append(merged)

    out = pd.concat(out, ignore_index=True)
    assert "code" in out.columns, "[merge_fundamentals] 缺少 code 列"
    return out

def industry_neutralize(df: pd.DataFrame, factor_cols: List[str], industry_col: str = "industry"):
    """
    每个时点、每个行业做去均值：x_neu = x - mean(x | datetime, industry)
    返回 (df, 新列名列表)
    """
    if industry_col not in df.columns:
        return df, list(factor_cols)

    df[industry_col] = df[industry_col].astype(str).fillna("UNKNOWN")

    out_cols = []
    for col in factor_cols:
        newc = f"{col}_neu"
        df[newc] = df.groupby(["datetime", industry_col])[col].transform(lambda x: x - x.mean())
        out_cols.append(newc)
    return df, out_cols

def add_cross_section_zscores(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    """每个时点做截面 z-score"""
    for c in factor_cols:
        zc = f"{c}_z"
        df[zc] = df.groupby("datetime")[c].transform(cross_section_zscore)
    return df

# ---------- 分位 buckets 与换手估算 ----------
def select_quantile_buckets(df, score_col="score", q=5):
    """返回每期各分位的持仓代码集合（用于估算换手）。"""
    d = df.dropna(subset=["datetime","code",score_col]).copy()
    d["rank"] = d.groupby("datetime")[score_col].rank(method="first")
    d["bucket"] = d.groupby("datetime")["rank"].transform(
        lambda x: pd.qcut(x, q, labels=range(1,q+1))
    ).astype(int)
    buckets = (d.groupby(["datetime","bucket"])["code"]
                 .apply(lambda s: set(s.astype(str)))
                 .unstack("bucket"))
    buckets.columns = [f"Q{c}" for c in buckets.columns]
    buckets.index = pd.to_datetime(buckets.index)
    return buckets

def estimate_turnover(buckets_df: pd.DataFrame) -> pd.DataFrame:
    """对每个分位计算相邻期对称差/上期规模作为换手率。"""
    turns = {}
    for c in buckets_df.columns:
        prev = None
        vals = []
        for _, cur in buckets_df[c].items():
            if prev is None:
                vals.append(0.0)
            else:
                denom = len(prev) + 1e-12
                vals.append(len(prev.symmetric_difference(cur)) / denom)
            prev = cur
        turns[c] = vals
    out = pd.DataFrame(turns, index=buckets_df.index)
    return out

def ls_with_dynamic_cost(grp: pd.DataFrame, turns: pd.DataFrame, bps_per_side: float = 0.0008):
    """
    基于换手的动态成本扣减：
    每期净LS = LS - (turn(Q5)+turn(Q1)) * bps_per_side * 2
    返回：ls_net（逐 period），cum_net（按 period 复合后的累计）
    """
    # 与 grp 对齐
    turns = turns.reindex(grp.index).fillna(0.0)
    cost = (turns.get("Q5", 0.0) + turns.get("Q1", 0.0)) * bps_per_side * 2.0
    ls_net = grp["LS"] - cost
    cum_net = (1.0 + ls_net).cumprod() - 1.0
    return ls_net, cum_net

def compute_oos_r2_gkx(df: pd.DataFrame, score_col="score", ret_col="excess_ret_t1") -> float:
    """
    Gu-Kelly-Xiu 的横截面 OOS R²：以当期横截面均值作为基准。
    R2 = 1 - sum( (r - yhat)^2 ) / sum( (r - crosssec_mean_t)^2 )
    """
    te = df.dropna(subset=["datetime","code",score_col,ret_col]).copy()
    te["datetime"] = pd.to_datetime(te["datetime"])
    num = ((te[ret_col] - te[score_col]) ** 2).sum()
    cs_mean = te.groupby("datetime")[ret_col].transform("mean")
    den = ((te[ret_col] - cs_mean) ** 2).sum()
    return float(1.0 - num / (den + 1e-12))

def leak_check_ic_samebar(df: pd.DataFrame, score_col="score", samebar_ret_col="ret_i") -> float:
    """
    泄露自检：用打分与“同一期收益”（不是 t+1）做截面 Spearman IC。
    正常应接近 0；若显著为正，说明还有泄露风险。
    """
    d = df.dropna(subset=["datetime","code",score_col,samebar_ret_col]).copy()
    ic = (d.groupby("datetime")
            .apply(lambda g: spearmanr(g[score_col], g[samebar_ret_col]).correlation)
            .dropna())
    return float(ic.mean())


# ------------- 评估与图表 -------------
def plot_distribution(y: pd.Series, out_dir: str):
    if not CONFIG["PLOT"]:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(y.dropna(), bins=120, alpha=0.85)
    plt.title("Excess Return (t+1) Distribution")
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
    plt.figure(figsize=(10, 4))
    plt.plot(roll["datetime"], roll["vol_30"])
    plt.title("HS300 5-min Rolling Volatility (30 bars)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_index_rolling_vol.png"), dpi=300)
    plt.close()

def plot_xgb_importance(model, feature_names, out_dir: str):
    if not CONFIG["PLOT"]:
        return
    imp = pd.Series(getattr(model, "feature_importances_", None), index=feature_names)
    if imp.isnull().all():
        return
    imp = imp.sort_values()
    plt.figure(figsize=(8, 6))
    imp.plot(kind="barh")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_xgb_importance.png"), dpi=300)
    plt.close()

def plot_corr_heatmap(df: pd.DataFrame, feature_names: List[str], out_dir: str):
    if not CONFIG["PLOT"]:
        return
    sub = df[feature_names].dropna()
    if len(sub) == 0:
        return
    samp = sub.sample(min(50000, len(sub)), random_state=CONFIG["SEED"])
    corr = samp.corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_corr_heatmap.png"), dpi=300)
    plt.close()

def plot_dnn_history(history, out_dir: str):
    if not CONFIG["PLOT"]:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(history.history.get("loss", []), label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.title("DNN Learning Curve (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_dnn_learning_curve.png"), dpi=300)
    plt.close()

def plot_model_comparison(summary_json_path, out_dir):
    import json
    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    pref = ["lasso", "xgboost", "dnn", "ipca"]
    models, r2s, mses = [], [], []
    for m in pref:
        if m in summary and isinstance(summary[m], dict) and "R2" in summary[m]:
            models.append(m)
            r2s.append(float(summary[m]["R2"]))
            mses.append(float(summary[m]["MSE"]))
    if not models:
        print("[plot_model_comparison] no metrics")
        return

    plt.figure(figsize=(6, 4))
    plt.bar(models, r2s)
    plt.title("Model Comparison: R²")
    plt.ylabel("R²")
    for i, v in enumerate(r2s):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_model_r2_comparison.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(models, mses)
    plt.title("Model Comparison: MSE")
    plt.ylabel("MSE")
    for i, v in enumerate(mses):
        plt.text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_model_mse_comparison.png"), dpi=300)
    plt.close()

def compute_ic(df, score_col="score", ret_col="excess_ret_t1"):
    out = (
        df.dropna(subset=[score_col, ret_col, "datetime", "code"])
          .groupby("datetime")
          .apply(lambda g: spearmanr(g[score_col], g[ret_col]).correlation)
    )
    ic_series = out.dropna()
    stats = {
        "IC_mean": float(ic_series.mean()),
        "IC_std": float(ic_series.std(ddof=1)),
        "IC_IR": float(ic_series.mean() / (ic_series.std(ddof=1) + 1e-12)),
    }
    return ic_series, stats

def plot_ic(ic_series, out_dir):
    if not CONFIG["PLOT"]:
        return
    plt.figure(figsize=(10, 3.5))
    plt.plot(ic_series.index, ic_series.values)
    plt.axhline(0, ls="--")
    plt.title("Cross-sectional Spearman IC (per bar)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_ic_timeseries.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(ic_series.values, bins=50)
    plt.title("IC Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_ic_hist.png"), dpi=300)
    plt.close()

def quantile_longshort(df, score_col="score", ret_col="excess_ret_t1", q=5):
    d = df.dropna(subset=[score_col, ret_col, "datetime", "code"]).copy()
    d["bucket"] = d.groupby("datetime")[score_col].transform(
        lambda x: pd.qcut(x.rank(method="first"), q, labels=range(1, q + 1))
    ).astype(int)
    grp = (
        d.groupby(["datetime", "bucket"])[ret_col]
         .mean()
         .unstack("bucket")
         .sort_index()
    )
    grp.columns = [f"Q{c}" for c in grp.columns]
    grp["LS"] = grp[f"Q{q}"] - grp["Q1"]
    cum = (1 + grp).cumprod() - 1
    return grp, cum

def plot_quantiles(grp, cum, out_dir):
    if not CONFIG["PLOT"]:
        return
    plt.figure(figsize=(9, 4))
    for c in [c for c in grp.columns if c.startswith("Q") and c != "LS"]:
        plt.plot(grp.index, grp[c], label=c, alpha=0.7)
    plt.title("Quantile Portfolio Returns (per period)")
    plt.legend(ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_quantile_returns.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(cum.index, cum["LS"])
    plt.title("Long-Short (Q5-Q1) Cumulative Return (no cost)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_ls_cum.png"), dpi=300)
    plt.close()

def ls_with_cost(grp, est_turnover_bps=0.0002):
    """给定每期 LS 收益与一个简单的平均换手成本基点，估算净值曲线"""
    ls_net = grp["LS"] - est_turnover_bps
    cum_net = (1 + ls_net).cumprod() - 1
    return ls_net, cum_net

def resample_quantile_table(grp: pd.DataFrame, rule: str = "1D", how: str = "mean") -> pd.DataFrame:
    """
    对逐 period 的分位收益表（列含 Q1..Qq 和 LS）按时间重采样到更低频（如 '1D'）。
    how: 'mean' 表示对日内取均值（等权），也可传 'sum'/'median' 等。
    返回重采样后的同结构表。
    """
    if not isinstance(grp.index, pd.DatetimeIndex):
        idx = pd.to_datetime(grp.index)
        grp = grp.copy()
        grp.index = idx
    if how == "mean":
        daily = grp.resample(rule).mean()
    elif how == "sum":
        daily = grp.resample(rule).sum()
    else:
        daily = getattr(grp.resample(rule), how)()
    return daily

