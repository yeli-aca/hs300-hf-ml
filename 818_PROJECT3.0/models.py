# models.py
# Lasso / XGB / DNN + 时间切分、WF评估

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

from utils import CONFIG


# ------------------------
# 时间切分 & Walk-forward
# ------------------------

def time_split(df: pd.DataFrame, test_ratio=0.2):
    cutoff = df["datetime"].quantile(1 - test_ratio)
    train = df[df["datetime"] < cutoff].copy()
    test  = df[df["datetime"] >= cutoff].copy()
    return train, test, cutoff


# models.py
def time_folds(df, n_splits=4, embargo_bars=5):
    ts = df["datetime"].sort_values().unique()
    borders = np.linspace(0, len(ts), n_splits+1).astype(int)
    folds=[]
    for k in range(1, n_splits+1):
        train_end = ts[borders[k-1]:borders[k]].max()
        # embargo：训练末尾与测试开始之间留空窗
        test_start_idx = min(borders[k] + embargo_bars, len(ts)-1)
        test_end_idx   = min(test_start_idx + (borders[1]-borders[0]), len(ts)-1)
        test_end  = ts[test_end_idx]
        tr = df["datetime"] <= train_end
        te = (df["datetime"] > train_end) & (df["datetime"] <= test_end)
        if te.sum()==0 or tr.sum()==0: continue
        folds.append((df.index[tr], df.index[te]))
    return folds



def wf_eval_lasso(data: pd.DataFrame, use_cols: List[str], n_splits=4, seed=42):
    scores = []
    folds = time_folds(data, n_splits=n_splits)
    for i, (tr_idx, te_idx) in enumerate(folds, 1):
        X_tr, y_tr = data.loc[tr_idx, use_cols], data.loc[tr_idx, "excess_ret_t1"]
        X_te, y_te = data.loc[te_idx, use_cols], data.loc[te_idx, "excess_ret_t1"]
        model = LassoCV(cv=5, random_state=seed)
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        r2 = r2_score(y_te, y_hat)
        mse = mean_squared_error(y_te, y_hat)
        scores.append((r2, mse))
        print(f"[WF-Lasso] fold{i}: R2={r2:.6f} MSE={mse:.6e}")
    if scores:
        r2s, mses = np.array(scores).T
        print(f"[WF-Lasso] mean R2={r2s.mean():.6f} | mean MSE={mses.mean():.6e}")
    return scores


# ------------------------
# 模型：Lasso
# ------------------------

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

    # === SHAP（可选） ===
    if CONFIG.get("DO_SHAP", False):
        try:
            import shap
            bg = X_train.sample(min(2000, len(X_train)), random_state=CONFIG["SEED"])
            Xt = X_test.sample(min(10000, len(X_test)), random_state=CONFIG["SEED"])

            try:
                explainer = shap.LinearExplainer(model, bg, feature_dependence="independent")
                shap_values = explainer.shap_values(Xt)
            except Exception:
                bg_k = bg.sample(min(200, len(bg)), random_state=CONFIG["SEED"])
                explainer = shap.KernelExplainer(model.predict, bg_k)
                shap_values = explainer.shap_values(Xt, nsamples=100)

            shap.summary_plot(shap_values, Xt, show=False, plot_type="bar")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_lasso_shap_summary_bar.png"), dpi=300)
            plt.close()

            shap.summary_plot(shap_values, Xt, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_lasso_shap_summary.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"[SHAP-Lasso] skip: {e}")

    return {"r2": r2, "mse": mse, "model": model, "y_pred": y_pred}


# ------------------------
# 模型：XGBoost
# ------------------------

# models.py
def run_xgb(Xtr, ytr, Xte, yte, out_dir):
    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_squared_error
    from utils import plot_xgb_importance, CONFIG

    params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1e-3,
        reg_lambda=1.0,
        random_state=CONFIG["SEED"],
        tree_method="hist",
        n_jobs=-1,
        # 关键：把评估指标放到构造器里，而不是 fit()
        eval_metric="rmse",
    )
    model = xgb.XGBRegressor(**params)

    # 训练（兼容不同版本的 fit 签名）
    try:
        model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xte, yte)], verbose=False)
    except TypeError:
        try:
            model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xte, yte)])
        except TypeError:
            model.fit(Xtr, ytr)

    y_pred = model.predict(Xte)
    r2 = r2_score(yte, y_pred)
    mse = mean_squared_error(yte, y_pred)

    try:
        plot_xgb_importance(model, list(Xtr.columns), out_dir)
    except Exception as e:
        print("[PLOT] xgb importance skip:", e)

    print(f"[XGBoost] R2={r2:.6f}   MSE={mse:.6e}")
    return {"model": model, "y_pred": y_pred, "r2": r2, "mse": mse}


# ------------------------
# 模型：DNN
# ------------------------

def run_dnn(Xtr, ytr, Xte, yte, out_dir: str):
    model = Sequential([
        Dense(256, activation="relu", kernel_regularizer=l2(1e-4), input_shape=(Xtr.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation="relu", kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1, activation="linear")
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

    cb = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
    ]

    history = model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=50,
        batch_size=1024,
        verbose=2,
        callbacks=cb
    )

    y_pred = model.predict(Xte).ravel()
    r2 = r2_score(yte, y_pred)
    mse = mean_squared_error(yte, y_pred)
    print(f"[DNN] R2={r2:.6f}  MSE={mse:.6e}")

    # 保存学习曲线
    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.legend()
    plt.title("Optimized DNN Learning Curve")
    plt.savefig(os.path.join(out_dir, "fig_dnn_learning_curve_opt.png"), dpi=300)
    plt.close()

    # === SHAP（可选） ===
    if CONFIG.get("DO_SHAP", False):
        try:
            import shap
            bg = Xtr.sample(min(1000, len(Xtr)), random_state=CONFIG["SEED"]).values
            Xt = Xte.sample(min(3000, len(Xte)), random_state=CONFIG["SEED"]).values

            shap_values = None
            try:
                explainer = shap.DeepExplainer(model, bg)
                shap_values = explainer.shap_values(Xt)
            except Exception:
                try:
                    explainer = shap.GradientExplainer(model, bg)
                    shap_values = explainer.shap_values(Xt)
                except Exception:
                    bg_k = Xtr.sample(min(200, len(Xtr)), random_state=CONFIG["SEED"]).values
                    Xt_k = Xte.sample(min(1000, len(Xte)), random_state=CONFIG["SEED"]).values
                    explainer = shap.KernelExplainer(lambda z: model.predict(z, batch_size=128).flatten(), bg_k)
                    shap_values = explainer.shap_values(Xt_k, nsamples=100)
                    Xt = Xt_k

            if isinstance(shap_values, list) and len(shap_values) == 1:
                shap_values = shap_values[0]

            shap.summary_plot(shap_values, Xt, show=False, plot_type="bar", feature_names=list(Xtr.columns))
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_dnn_shap_summary_bar.png"), dpi=300)
            plt.close()

            shap.summary_plot(shap_values, Xt, show=False, feature_names=list(Xtr.columns))
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_dnn_shap_summary.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"[SHAP-DNN] skip: {e}")

    return {"r2": r2, "mse": mse, "model": model, "y_pred": y_pred}
