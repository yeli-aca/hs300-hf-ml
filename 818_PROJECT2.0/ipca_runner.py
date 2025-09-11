# ipca_runner.py
# IPCA：build_ipca_data + in-sample评分 + train/test式 OOS评分
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from ipca_core import InstrumentedPCA  # 你的实现

def build_ipca_data(df, factor_cols):
    """从原始 DataFrame 构造 IPCA 输入矩阵"""
    need_cols = ["datetime", "code", "excess_ret_t1"] + list(factor_cols)
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise KeyError(f"[build_ipca_data] missing columns: {miss}")

    d = df[need_cols].copy()
    d["datetime"] = pd.to_datetime(d["datetime"], utc=False, errors="coerce")
    d["code"] = d["code"].astype(str)

    # 删除缺失/非有限值
    d = d.dropna(subset=["datetime", "code", "excess_ret_t1"] + list(factor_cols))
    d = d[np.isfinite(d["excess_ret_t1"].astype(float))]
    for c in factor_cols:
        d = d[np.isfinite(d[c].astype(float))]

    if d.empty:
        raise ValueError("[build_ipca_data] after dropna no data left.")

    # 输出矩阵
    Y = d["excess_ret_t1"].astype(float).to_numpy()
    X = d[factor_cols].astype(float).to_numpy()
    dates_list = d["datetime"].to_numpy()
    codes_list = d["code"].to_numpy()
    fac_names = list(factor_cols)

    print(f"[DEBUG] build_ipca_data: {d['datetime'].nunique()} dates, {d['code'].nunique()} entities, {len(d)} rows")
    return Y, X, dates_list, codes_list, fac_names


def run_ipca_insample(Y, X, dates, codes, fac_names, out_dir, n_factors=3):
    """IPCA in-sample 拟合与评估"""
    Xp, yp, mi = _flat_panel(Y, X, dates, codes, fac_names)
    Xp.index, yp.index = mi, mi

    common_idx = Xp.index.intersection(yp.index)
    Xp, yp = Xp.loc[common_idx], yp.loc[common_idx]

    model = InstrumentedPCA(n_factors=n_factors, intercept=True)
    model.fit(Xp, yp)

    # 保存因子载荷与时间序列
    Gamma, Factors = model.get_factors(label_ind=True)
    Gamma.to_csv(os.path.join(out_dir, "ipca_gamma.csv"))
    Factors.to_csv(os.path.join(out_dir, "ipca_factors.csv"))

    # 预测与评估
    y_pred = model.predict(Xp, indices=Xp.index.to_frame().values, data_type="panel")
    r2  = r2_score(yp.values, y_pred)
    mse = mean_squared_error(yp.values, y_pred)
    print(f"[IPCA-IS] R2={r2:.6f}  MSE={mse:.6e}")

    pd.DataFrame({"y_true": yp.values, "y_pred": y_pred}, index=yp.index).to_csv(
        os.path.join(out_dir, "ipca_insample_pred.csv")
    )

    return {"r2": r2, "mse": mse, "model": model, "Gamma": Gamma,
            "Factors": Factors, "y_pred": y_pred}


def run_ipca_oos_by_split(train_df, test_df, use_cols, out_dir, n_factors=3):
    """IPCA out-of-sample 训练 + 测试"""
    # ---- Train ----
    Y_tr, X_tr, dates_tr, codes_tr, fac_tr = build_ipca_data(train_df, use_cols)
    Xp_tr, yp_tr, mi_tr = _flat_panel(Y_tr, X_tr, dates_tr, codes_tr, fac_tr)
    Xp_tr.index, yp_tr.index = mi_tr, mi_tr
    common_idx = Xp_tr.index.intersection(yp_tr.index)
    Xp_tr, yp_tr = Xp_tr.loc[common_idx], yp_tr.loc[common_idx]

    model = InstrumentedPCA(n_factors=n_factors, intercept=True)
    model.fit(Xp_tr, yp_tr)

    # ---- Test ----
    Y_te, X_te, dates_te, codes_te, fac_te = build_ipca_data(test_df, use_cols)
    Xp_te, yp_te, mi_te = _flat_panel(Y_te, X_te, dates_te, codes_te, fac_te)
    Xp_te.index, yp_te.index = mi_te, mi_te
    common_idx = Xp_te.index.intersection(yp_te.index)
    Xp_te, yp_te = Xp_te.loc[common_idx], yp_te.loc[common_idx]

    try:
        y_pred = model.predict(Xp_te, indices=Xp_te.index.to_frame().values, data_type="panel")
    except Exception:
        y_pred = model.predict(Xp_te, indices=Xp_te.index.to_frame().values,
                               data_type="panel", mean_factor=True)

    r2  = r2_score(yp_te.values, y_pred)
    mse = mean_squared_error(yp_te.values, y_pred)
    print(f"[IPCA-OOS] R2={r2:.6f}  MSE={mse:.6e}")

    pd.DataFrame({"y_true": yp_te.values, "y_pred": y_pred}, index=yp_te.index).to_csv(
        os.path.join(out_dir, "ipca_oos_pred.csv")
    )

    return {"r2": r2, "mse": mse, "model": model, "y_pred": y_pred}


def _flat_panel(Y, X, dates, codes, fac_names):
    """把 (Y, X, 日期, 股票代码) 转换成带 MultiIndex 的面板形式"""
    Xp = pd.DataFrame(np.asarray(X, float), columns=list(fac_names))
    yp = pd.Series(np.asarray(Y, float), name="target")

    # 屏蔽非有限值
    mask = np.isfinite(yp.values) & np.all(np.isfinite(Xp.values), axis=1)
    Xp = Xp.loc[mask].reset_index(drop=True)
    yp = yp.loc[mask].reset_index(drop=True)

    dates = np.asarray(dates)[mask]
    codes = np.asarray(codes)[mask]
    mi = pd.MultiIndex.from_arrays([dates, codes], names=["datetime", "code"])

    if Xp.shape[0] != yp.shape[0] or Xp.shape[0] != mi.shape[0]:
        raise ValueError("[_flat_panel] shape mismatch among X, y, and indices.")

    return Xp, yp, mi
