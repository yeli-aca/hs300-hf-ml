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
    plot_model_comparison
)
from models import time_split, wf_eval_lasso, run_lasso, run_xgb, run_dnn
from ipca_runner import build_ipca_data, run_ipca_insample, run_ipca_oos_by_split


def main():
    ensure_outdir(CONFIG["OUT_DIR"])

    # ---- 数据 ----
    panel = load_stock_panel(CONFIG["STOCK_DIR"], CONFIG["ROW_SAMPLE_FRAC"])
    idx    = load_index_5min(CONFIG["INDEX_FILE"])
    fund   = load_fundamentals(CONFIG["FUND_FILE"])

    print("🟢 原始面板：", len(panel), "rows,", panel["code"].nunique(), "stocks",
          f"range=[{panel['datetime'].min()} ~ {panel['datetime'].max()}]")

    # ---- 特征与目标 ----
    df, factor_cols = build_features(panel)
    print("[DEBUG] after build_features:", df.columns)
    del panel; gc.collect()

    df = merge_index_and_target(df, idx)
    print("[DEBUG] after merge_index_and_target:", df.columns)

    if fund is not None:
        df = merge_fundamentals(df, fund)
        print("[DEBUG] after merge_fundamentals:", df.columns)

    df, eff_cols = industry_neutralize(df, factor_cols, "industry")
    print("[DEBUG] after industry_neutralize:", df.columns)

    df = add_cross_section_zscores(df, eff_cols)
    print("[DEBUG] after add_cross_section_zscores:", df.columns)

    use_cols = [c+"_z" for c in eff_cols]

    # ---- 在 expected 子集前后检查 ----
    print("[DEBUG] before expected filter:", df.columns)
    expected = ["datetime", "excess_ret_t1"] + use_cols
    if "code" in df.columns:
        expected.insert(1, "code")   # 动态保留 code
    data = df[[c for c in expected if c in df.columns]].dropna()
    print("[DEBUG] after expected filter (data):", data.columns)

    must_have = ["datetime", "excess_ret_t1"]
    missing_min = [c for c in must_have if c not in data.columns]
    if missing_min:
        raise ValueError(f"关键列缺失: {missing_min}")

    if "code" not in data.columns and "code" in df.columns:
        data = data.merge(df[["datetime", "code"]], on="datetime", how="left")
        print("[DEBUG]补回 code 列:", data.columns)

    data = reduce_mem(data)

    # ---- 简要图表 ----
    plot_distribution(data["excess_ret_t1"], CONFIG["OUT_DIR"])
    plot_corr_heatmap(data, use_cols, CONFIG["OUT_DIR"])
    plot_index_vol(idx, CONFIG["OUT_DIR"])

    # ---- Walk-forward（可选）----
    try:
        _ = wf_eval_lasso(data[["datetime", "excess_ret_t1"] + use_cols], use_cols, n_splits=4)
    except Exception as e:
        print("[WF] skip:", e)

    # ---- Train/Test ----
    train, test, cutoff = time_split(data, CONFIG["TEST_RATIO"])
    print("[DEBUG] train columns:", train.columns)
    print("[DEBUG] test columns:", test.columns)

    Xtr, ytr = train[use_cols], train["excess_ret_t1"]
    Xte, yte = test[use_cols],  test["excess_ret_t1"]

    # ---- 三模型 ----
    res_lasso = run_lasso(Xtr, ytr, Xte, yte, CONFIG["OUT_DIR"])
    res_xgb   = run_xgb  (Xtr, ytr, Xte, yte, CONFIG["OUT_DIR"])
    res_dnn   = run_dnn  (Xtr, ytr, Xte, yte, CONFIG["OUT_DIR"])

    # ---- XGB 的 IC / 分位 / 成本敏感 ----
    base_cols = safe_cols(test, ["datetime", "code", "excess_ret_t1"])
    test_with_score = test[base_cols].copy()
    if "code" not in test_with_score.columns and "code" in df.columns:
        test_with_score = test_with_score.merge(df[["datetime","code"]], on="datetime", how="left")

    test_with_score["score"] = res_xgb["y_pred"]

    print("[DEBUG] columns in test_with_score:", list(test_with_score.columns))
    ic_series, ic_stats = compute_ic(test_with_score, "score", "excess_ret_t1")
    plot_ic(ic_series, CONFIG["OUT_DIR"])

    grp, cum = quantile_longshort(test_with_score, "score", "excess_ret_t1", q=5)
    plot_quantiles(grp, cum, CONFIG["OUT_DIR"])
    grp.mean().to_csv(os.path.join(CONFIG["OUT_DIR"], "quantile_mean_returns.csv"))

    # 成本敏感多空
    ls_net, cum_net = ls_with_cost(grp, est_turnover_bps=0.0002)
    plt.figure(figsize=(9, 4))
    plt.plot(cum_net.index, cum_net.values)
    plt.title("Long-Short (Q5-Q1) Cumulative Return with Cost (2 bps)")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["OUT_DIR"], "fig_ls_cum_with_cost.png"), dpi=300)
    plt.close()

        # ---- IPCA ----
    res_ipca = run_ipca_oos_by_split(train, test, use_cols, CONFIG["OUT_DIR"], n_factors=3)

    # ---- 保存 IPCA 结果额外提示 ----
    print("[INFO] IPCA Gamma, Factors, and prediction files saved to:", CONFIG["OUT_DIR"])


    # ---- 汇总并出图 ----
    summary = {
        "lasso":  {"R2": float(res_lasso["r2"]), "MSE": float(res_lasso["mse"])},
        "xgboost":{"R2": float(res_xgb["r2"]),   "MSE": float(res_xgb["mse"])},
        "dnn":    {"R2": float(res_dnn["r2"]),   "MSE": float(res_dnn["mse"])},
        "ipca":   {"R2": float(res_ipca["r2"]),  "MSE": float(res_ipca["mse"])},
        "cutoff": str(cutoff),
        "n_train": int(len(train)),
        "n_test":  int(len(test)),
        "n_codes": int(data["code"].nunique()) if "code" in data.columns else None
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
