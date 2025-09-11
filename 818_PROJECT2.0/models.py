# models.py
# Lasso / XGB / DNN + 时间切分、WF评估
import os
import matplotlib.pyplot as plt

import json
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LassoCV

import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils import CONFIG, plot_xgb_importance, plot_dnn_history

def time_split(df: pd.DataFrame, test_ratio=0.2):
    cutoff = df["datetime"].quantile(1 - test_ratio)
    train = df[df["datetime"] < cutoff].copy()
    test  = df[df["datetime"] >= cutoff].copy()
    return train, test, cutoff

def time_folds(df, n_splits=4):
    ts = df["datetime"].sort_values().unique()
    borders = np.linspace(0, len(ts), n_splits+1).astype(int)
    folds=[]
    for k in range(1, n_splits+1):
        train_end = ts[borders[k-1]:borders[k]].max()
        test_end  = ts[min(borders[k]+max(1,(borders[1]-borders[0])//4), len(ts)-1)]
        tr = df["datetime"] <= train_end
        te = (df["datetime"] > train_end) & (df["datetime"] <= test_end)
        if te.sum()==0 or tr.sum()==0: continue
        folds.append((df.index[tr], df.index[te]))
    return folds

def wf_eval_lasso(data, use_cols, n_splits=4, seed=42):
    scores=[]
    folds = time_folds(data, n_splits=n_splits)
    for i,(tr_idx, te_idx) in enumerate(folds,1):
        X_tr, y_tr = data.loc[tr_idx, use_cols], data.loc[tr_idx,"excess_ret_t1"]
        X_te, y_te = data.loc[te_idx, use_cols], data.loc[te_idx,"excess_ret_t1"]
        model = LassoCV(cv=5, random_state=seed)
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        r2 = r2_score(y_te, y_hat); mse = mean_squared_error(y_te, y_hat)
        scores.append((r2, mse))
        print(f"[WF-Lasso] fold{i}: R2={r2:.6f} MSE={mse:.6e}")
    if scores:
        r2s, mses = np.array(scores).T
        print(f"[WF-Lasso] mean R2={r2s.mean():.6f} | mean MSE={mses.mean():.6e}")
    return scores

def run_lasso(X_train, y_train, X_test, y_test, out_dir: str):
    model = LassoCV(cv=5, random_state=CONFIG["SEED"], n_jobs=None)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # 导出非零系数
    coef = pd.Series(model.coef_, index=X_train.columns)
    coef[coef != 0].sort_values(key=np.abs, ascending=False)\
        .to_csv(os.path.join(out_dir, "lasso_nonzero_coef.csv"))

    print(f"[Lasso] R2={r2:.6f}  MSE={mse:.6e}")

    # === SHAP（线性模型：LinearExplainer 最快；失败则退回 KernelExplainer，小样本） ===
    if CONFIG.get("DO_SHAP", False):
        try:
            import shap
            # 背景与解释样本（注意控制规模，避免过慢）
            bg = X_train.sample(min(2000, len(X_train)), random_state=CONFIG["SEED"])
            Xt = X_test.sample(min(10000, len(X_test)), random_state=CONFIG["SEED"])

            try:
                explainer = shap.LinearExplainer(model, bg, feature_dependence="independent")
                shap_values = explainer.shap_values(Xt)
            except Exception:
                # 退回 KernelExplainer（更慢）：再缩小样本与 nsamples
                bg_k = bg.sample(min(200, len(bg)), random_state=CONFIG["SEED"])
                explainer = shap.KernelExplainer(model.predict, bg_k)
                shap_values = explainer.shap_values(Xt, nsamples=100)

            # 条形图（全局重要性）
            shap.summary_plot(shap_values, Xt, show=False, plot_type="bar")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_lasso_shap_summary_bar.png"), dpi=300)
            plt.close()

            # 蜂群图（方向性 + 分布）
            shap.summary_plot(shap_values, Xt, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_lasso_shap_summary.png"), dpi=300)
            plt.close()

        except Exception as e:
            print(f"[SHAP-Lasso] skip: {e}")

    return {"r2": r2, "mse": mse, "model": model, "y_pred": y_pred}


def run_xgb(X_train, y_train, X_test, y_test, out_dir: str):
    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=CONFIG["SEED"], n_jobs=-1, reg_lambda=1.0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"[XGBoost] R2={r2:.6f}  MSE={mse:.6e}")
    plot_xgb_importance(model, list(X_train.columns), out_dir)
    return {"r2": r2, "mse": mse, "model": model, "y_pred": y_pred}

def run_dnn(X_train, y_train, X_test, y_test, out_dir: str):
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

    es   = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    rl   = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

    hist = model.fit(
        X_train, y_train,
        epochs=CONFIG["DNN_EPOCHS"],
        batch_size=CONFIG["DNN_BATCH"],
        validation_split=0.2,
        callbacks=[es, rl],
        verbose=1
    )

    y_pred = model.predict(X_test, batch_size=CONFIG["DNN_BATCH"]).flatten()
    r2  = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"[DNN] R2={r2:.6f}  MSE={mse:.6e}")

    # 学习曲线
    plot_dnn_history(hist, out_dir)

    # === SHAP for Keras：优先 DeepExplainer/GradientExplainer，失败再退 KernelExplainer ===
    if CONFIG.get("DO_SHAP", False):
        try:
            import shap
            # 采样（DNN 的 SHAP 很吃算力，强烈建议控制规模）
            bg = X_train.sample(min(1000, len(X_train)), random_state=CONFIG["SEED"]).values
            Xt = X_test.sample(min(3000, len(X_test)),  random_state=CONFIG["SEED"]).values

            shap_values = None
            # 1) DeepExplainer（优先，最准确；有时会因 TF/ops 不兼容失败）
            try:
                explainer = shap.DeepExplainer(model, bg)
                shap_values = explainer.shap_values(Xt)
            except Exception:
                # 2) GradientExplainer（更稳妥）
                try:
                    explainer = shap.GradientExplainer(model, bg)
                    shap_values = explainer.shap_values(Xt)
                except Exception:
                    # 3) KernelExplainer（通用但很慢）—再大幅降样本
                    bg_k = X_train.sample(min(200, len(X_train)), random_state=CONFIG["SEED"]).values
                    Xt_k = X_test.sample(min(1000, len(X_test)),  random_state=CONFIG["SEED"]).values
                    explainer = shap.KernelExplainer(lambda z: model.predict(z, batch_size=CONFIG["DNN_BATCH"]).flatten(), bg_k)
                    shap_values = explainer.shap_values(Xt_k, nsamples=100)
                    Xt = Xt_k  # 与 shap_values 对齐

            # shap_values 对于单输出是 list(len=1)
            if isinstance(shap_values, list) and len(shap_values) == 1:
                shap_values = shap_values[0]

            # 画图
            shap.summary_plot(shap_values, Xt, show=False, plot_type="bar", feature_names=list(X_train.columns))
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_dnn_shap_summary_bar.png"), dpi=300)
            plt.close()

            shap.summary_plot(shap_values, Xt, show=False, feature_names=list(X_train.columns))
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_dnn_shap_summary.png"), dpi=300)
            plt.close()

        except Exception as e:
            print(f"[SHAP-DNN] skip: {e}")

    return {"r2": r2, "mse": mse, "model": model, "y_pred": y_pred}
