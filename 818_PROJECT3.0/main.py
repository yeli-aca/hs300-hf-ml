# main.py
# 入口：串起来跑 & 输出指标与图表
import os, json, gc
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    CONFIG, ensure_outdir, load_stock_panel, load_index_5min, load_fundamentals,
    build_features, merge_index_and_target, merge_fundamentals,
    industry_neutralize, add_cross_section_zscores, reduce_mem, safe_cols,
    plot_distribution, plot_corr_heatmap, plot_index_vol,
    compute_ic, plot_ic, quantile_longshort, plot_quantiles, ls_with_cost,
    plot_model_comparison, select_quantile_buckets, estimate_turnover
)
from models import time_split, wf_eval_lasso, run_lasso, run_xgb, run_dnn
from ipca_runner import run_ipca_oos_by_split


def winsorize_series(s, limits=(-5, 5)):
    return s.clip(lower=limits[0], upper=limits[1])


def ensure_code(df: pd.DataFrame, stage: str):
    """统一检查 df 是否包含 code 列"""
    if "code" not in df.columns:
        raise ValueError(f"[{stage}] 缺少 code 列！实际列={df.columns.tolist()}")


def main():
    ensure_outdir(CONFIG["OUT_DIR"])

    # ---- 数据 ----
    panel = load_stock_panel(CONFIG["STOCK_DIR"], CONFIG["ROW_SAMPLE_FRAC"])
    idx    = load_index_5min(CONFIG["INDEX_FILE"])
    fund   = load_fundamentals(CONFIG["FUND_FILE"])

    # 基本健康检查
    for obj_name, obj in [("panel", panel), ("idx", idx)]:
        if obj is None or len(obj) == 0:
            raise ValueError(f"{obj_name} 为空，请检查数据输入。")
    ensure_code(panel, "raw panel")

    print("🟢 原始面板：", len(panel), "rows,", panel["code"].nunique(), "stocks",
          f"range=[{panel['datetime'].min()} ~ {panel['datetime'].max()}]")

    # ---- 特征与目标（公共基础）----
    df_base, factor_cols = build_features(panel)
    del panel; gc.collect()
    ensure_code(df_base, "build_features")

    df_base = merge_index_and_target(df_base, idx)
    ensure_code(df_base, "merge_index_and_target")

    if fund is not None:
        df_base = merge_fundamentals(df_base, fund)
        ensure_code(df_base, "merge_fundamentals")

    # ========== ① IPCA 专用支路：不做行业中性化，仅做“截面 zscore + winsor” ==========
    df_ipca = df_base[["datetime", "code", "excess_ret_t1"] + factor_cols].copy()
    df_ipca = add_cross_section_zscores(df_ipca, factor_cols)
    use_cols_ipca = [c + "_z" for c in factor_cols if f"{c}_z" in df_ipca.columns]
    if not use_cols_ipca:
        raise ValueError("IPCA use_cols_ipca 为空，请检查因子列。")
    for c in use_cols_ipca:
        df_ipca[c] = winsorize_series(df_ipca[c], limits=(-5, 5))
    data_ipca = df_ipca[["datetime", "code", "excess_ret_t1"] + use_cols_ipca]\
                    .dropna(subset=["excess_ret_t1"] + use_cols_ipca)\
                    .copy()

    # ========== ② ML 支路（Lasso/XGB/DNN）：行业中性化 + 截面 zscore + winsor ==========
    df_ml, eff_cols = industry_neutralize(df_base.copy(), factor_cols, "industry")
    df_ml = add_cross_section_zscores(df_ml, eff_cols)
    use_cols = [c + "_z" for c in eff_cols if f"{c}_z" in df_ml.columns]
    if not use_cols:
        raise ValueError("ML use_cols 为空，请检查行业中性化后的因子列。")
    for c in use_cols:
        df_ml[c] = winsorize_series(df_ml[c], limits=(-5, 5))

    expected = ["datetime", "code", "excess_ret_t1"] + use_cols
    data = df_ml[expected].dropna(subset=["excess_ret_t1"] + use_cols).copy()
    ensure_code(data, "data after filter")
    data = reduce_mem(data)

    # ---- 简要图表（基于 ML 支路的数据）----
    try: plot_distribution(data["excess_ret_t1"], CONFIG["OUT_DIR"])
    except Exception as e: print("[PLOT] distribution skip:", e)
    try: plot_corr_heatmap(data, use_cols, CONFIG["OUT_DIR"])
    except Exception as e: print("[PLOT] corr_heatmap skip:", e)
    try: plot_index_vol(idx, CONFIG["OUT_DIR"])
    except Exception as e: print("[PLOT] index_vol skip:", e)

    # ---- Walk-forward（可选）----
    try:
        _ = wf_eval_lasso(data[["datetime", "excess_ret_t1"] + use_cols], use_cols, n_splits=4)
    except Exception as e:
        print("[WF] skip:", e)

    # ---- Train/Test（ML 支路）----
    train, test, cutoff = time_split(data, CONFIG["TEST_RATIO"])
    ensure_code(train, "train"); ensure_code(test, "test")

    Xtr, ytr = train[use_cols], train["excess_ret_t1"]
    Xte, yte = test[use_cols],  test["excess_ret_t1"]

    # ---- 三模型 ----
    res_lasso = run_lasso(Xtr, ytr, Xte, yte, CONFIG["OUT_DIR"])
    res_xgb   = run_xgb  (Xtr, ytr, Xte, yte, CONFIG["OUT_DIR"])
    res_dnn   = run_dnn  (Xtr, ytr, Xte, yte, CONFIG["OUT_DIR"])

    # ---- XGB 的 IC / 分位 / 成本敏感 ----
    base_cols = safe_cols(test, ["datetime", "code", "excess_ret_t1"])
    test_with_score = test[base_cols].copy()
    ensure_code(test_with_score, "test_with_score")
    test_with_score["score"] = res_xgb["y_pred"]

    ic_series, ic_stats = compute_ic(test_with_score, "score", "excess_ret_t1")
    plot_ic(ic_series, CONFIG["OUT_DIR"])

    grp, cum = quantile_longshort(test_with_score, "score", "excess_ret_t1", q=5)
    plot_quantiles(grp, cum, CONFIG["OUT_DIR"])
    try: grp.mean().to_csv(os.path.join(CONFIG["OUT_DIR"], "quantile_mean_returns.csv"))
    except Exception as e: print("[SAVE] quantile_mean_returns skip:", e)

    # 成本敏感（常数成本演示）
    try:
        ls_net, cum_net = ls_with_cost(grp, est_turnover_bps=0.0002)
        plt.figure(figsize=(9, 4))
        plt.plot(cum_net.index, cum_net.values)
        plt.title("Long-Short (Q5-Q1) Cumulative Return with Cost (2 bps)")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["OUT_DIR"], "fig_ls_cum_with_cost.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("[PLOT] ls_with_cost skip:", e)

    # 动态换手成本（基于持仓变动）
    try:
        buckets = select_quantile_buckets(test_with_score, "score", q=5)
        turns   = estimate_turnover(buckets)   # index: datetime, cols: Q1..Q5
        cost_one_way = 0.0008  # 8 bps 单边
        ls_net_dyn   = grp["LS"] - (turns["Q5"] + turns["Q1"]) * cost_one_way
        cum_net_dyn  = (1 + ls_net_dyn).cumprod() - 1
        plt.figure(figsize=(9,4))
        plt.plot(cum_net_dyn.index, cum_net_dyn.values)
        plt.title("Long-Short (Q5-Q1) Cumulative Return — turnover-cost")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["OUT_DIR"], "fig_ls_cum_with_turnover_cost.png"), dpi=300)
        plt.close()
        turns.to_csv(os.path.join(CONFIG["OUT_DIR"], "turnover_by_quantile.csv"))
        ls_net_dyn.rename("LS_net_turnover_cost").to_csv(
            os.path.join(CONFIG["OUT_DIR"], "ls_net_turnover_cost.csv")
        )
    except Exception as e:
        print("[TURNOVER] skip:", e)

    # ---- IPCA OOS：选最优因子数（独立于 ML 支路）----
    train_ipca, test_ipca, _ = time_split(data_ipca, CONFIG["TEST_RATIO"])
    best_ipca, best_r2 = None, -1e9
    for k in [3, 5, 10]:
        try:
            res_ipca = run_ipca_oos_by_split(train_ipca, test_ipca, use_cols_ipca, CONFIG["OUT_DIR"], n_factors=k)
            if res_ipca["r2"] > best_r2:
                best_ipca, best_r2 = res_ipca, res_ipca["r2"]
        except Exception as e:
            print(f"[IPCA] n_factors={k} failed:", e)
    if best_ipca is None:
        raise RuntimeError("IPCA 全部配置失败")

    print(f"[IPCA] best result with n_factors={best_ipca['model'].n_factors}, R2={best_ipca['r2']:.6f}")
    print("[INFO] IPCA Gamma, Factors, and prediction files saved to:", CONFIG["OUT_DIR"])

    # ---- 汇总并出图 ----
    summary = {
        "lasso":  {"R2": float(res_lasso["r2"]), "MSE": float(res_lasso["mse"])},
        "xgboost":{"R2": float(res_xgb["r2"]),   "MSE": float(res_xgb["mse"])},
        "dnn":    {"R2": float(res_dnn["r2"]),   "MSE": float(res_dnn["mse"])},
        "ipca":   {"R2": float(best_ipca["r2"]), "MSE": float(best_ipca.get("mse", float('nan')))},
        "cutoff": str(cutoff),
        "n_train": int(len(train)),
        "n_test":  int(len(test)),
        "n_codes": int(data["code"].nunique())
    }
    with open(os.path.join(CONFIG["OUT_DIR"], "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[INFO] All results saved under: {CONFIG['OUT_DIR']}")

    plot_model_comparison(
        summary_json_path=os.path.join(CONFIG["OUT_DIR"], "metrics_summary.json"),
        out_dir=CONFIG["OUT_DIR"]
    )


if __name__ == "__main__":
    print(">>> starting main() ...")
    try:
        main()
    except Exception as e:
        print("[ERROR] main failed:", e)
    finally:
        print(">>> finished main()")
