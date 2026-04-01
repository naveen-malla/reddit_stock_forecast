"""
src/models.py
─────────────
Trains and evaluates three models:

  1. Naive Baseline  — yesterday's return as prediction (random-walk benchmark)
  2. XGBoost         — gradient-boosted trees
  3. LightGBM        — leaf-wise gradient boosting

FIXES vs v1:
  - Added naive baseline model (ChatGPT correctly flagged this gap)
  - LightGBM early stopping uses VALIDATION set (not test set — was data leakage)
  - XGBoost eval_set also uses validation set
  - Directional accuracy added as classification metric
  - Qualitative analysis improved

Evaluation metrics:
  MAE  — Mean Absolute Error
  RMSE — Root Mean Squared Error
  DA   — Directional Accuracy (% days where sign of prediction matches actual)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct = np.sign(y_pred) == np.sign(y_true)
    return float(correct.mean())


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    da = directional_accuracy(y_true, y_pred)
    return {"model": name, "MAE": mae, "RMSE": rmse, "DirectionalAccuracy": da}


def build_xgboost() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=cfg.random_state,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
        early_stopping_rounds=50,
    )


def build_lightgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=cfg.random_state,
        n_jobs=-1,
        verbose=-1,
    )


class NaiveBaseline:
    """
    Yesterday's return = tomorrow's predicted return.
    This is the simplest possible model — ML must beat this to be useful.
    In finance this is the random-walk null hypothesis.
    """

    def fit(self, X_train, y_train, **kwargs):
        # No training needed; we track the last known return externally
        return self

    def predict(self, X_test):
        # X_test has ret_1d as a feature — find its column index
        ret_idx = getattr(self, "_ret_col_idx", None)
        if ret_idx is not None:
            return X_test[:, ret_idx]
        # Fallback: predict zero (mean-reversion baseline)
        return np.zeros(len(X_test))

    def set_ret_col_idx(self, feature_cols: List[str]):
        try:
            self._ret_col_idx = feature_cols.index("ret_1d")
        except ValueError:
            self._ret_col_idx = None


class ModelTrainer:
    """Trains, evaluates, and persists all models."""

    def __init__(self):
        self.trained_models: Dict[str, object] = {}
        self.results: List[Dict] = []

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Train all models, evaluate, return comparison DataFrame."""
        self.results = []

        # ── 1. Naive Baseline ─────────────────────────────────────────────────
        logger.info("Evaluating naive baseline (yesterday's return) …")
        baseline = NaiveBaseline()
        baseline.set_ret_col_idx(feature_cols)
        baseline.fit(X_train, y_train)
        y_pred_baseline = baseline.predict(X_test)
        baseline_metrics = evaluate("Naive Baseline", y_test, y_pred_baseline)
        baseline_metrics["y_pred"] = y_pred_baseline
        self.results.append(baseline_metrics)
        self.trained_models["Naive Baseline"] = baseline
        logger.info(
            f"  Naive Baseline — MAE={baseline_metrics['MAE']:.5f}  "
            f"DA={baseline_metrics['DirectionalAccuracy']:.3f}"
        )

        # ── 2. XGBoost ────────────────────────────────────────────────────────
        logger.info("Training XGBoost …")
        xgb_model = build_xgboost()
        # FIX: use validation set for early stopping, NOT test set
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred_xgb = xgb_model.predict(X_test)
        xgb_metrics = evaluate("XGBoost", y_test, y_pred_xgb)
        xgb_metrics["y_pred"] = y_pred_xgb
        self.results.append(xgb_metrics)
        self.trained_models["XGBoost"] = xgb_model
        self._save_model("XGBoost", xgb_model, feature_cols)
        logger.info(
            f"  XGBoost — MAE={xgb_metrics['MAE']:.5f}  "
            f"RMSE={xgb_metrics['RMSE']:.5f}  DA={xgb_metrics['DirectionalAccuracy']:.3f}"
        )

        # ── 3. LightGBM ───────────────────────────────────────────────────────
        logger.info("Training LightGBM …")
        lgb_model = build_lightgbm()
        # FIX: use validation set for early stopping, NOT test set
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        y_pred_lgb = lgb_model.predict(X_test)
        lgb_metrics = evaluate("LightGBM", y_test, y_pred_lgb)
        lgb_metrics["y_pred"] = y_pred_lgb
        self.results.append(lgb_metrics)
        self.trained_models["LightGBM"] = lgb_model
        self._save_model("LightGBM", lgb_model, feature_cols)
        logger.info(
            f"  LightGBM — MAE={lgb_metrics['MAE']:.5f}  "
            f"RMSE={lgb_metrics['RMSE']:.5f}  DA={lgb_metrics['DirectionalAccuracy']:.3f}"
        )

        return self._comparison_df()

    def print_comparison(self, results_df: pd.DataFrame | None = None) -> None:
        df = results_df if results_df is not None else self._comparison_df()
        print("\n" + "═" * 70)
        print("  MODEL PERFORMANCE COMPARISON")
        print("═" * 70)
        print(f"  {'Model':<18} {'MAE':>10} {'RMSE':>10} {'Dir. Acc.':>12}  {'vs Baseline':>12}")
        print("─" * 70)

        baseline_da = df[df["model"] == "Naive Baseline"]["DirectionalAccuracy"].values
        baseline_da = baseline_da[0] if len(baseline_da) else 0.5

        for _, row in df.iterrows():
            vs = ""
            if row["model"] != "Naive Baseline":
                diff = row["DirectionalAccuracy"] - baseline_da
                vs = f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}"
            print(
                f"  {row['model']:<18} {row['MAE']:>10.5f} {row['RMSE']:>10.5f} "
                f"{row['DirectionalAccuracy']:>11.1%}  {vs:>12}"
            )
        print("─" * 70)

        best_da_row = df[df["model"] != "Naive Baseline"]["DirectionalAccuracy"].idxmax()
        best_mae_row = df[df["model"] != "Naive Baseline"]["MAE"].idxmin()
        print(f"  Best MAE model:        {df.loc[best_mae_row, 'model']}")
        print(f"  Best Directional Acc:  {df.loc[best_da_row, 'model']}")
        print("═" * 70)

        print("\nLIMITATIONS & CONTEXT")
        print("─" * 70)
        print("  • Directional accuracy > 50% does NOT guarantee profitability.")
        print("  • Reddit sentiment may reflect noise, memes, or coordinated posts.")
        print("  • Past predictability does not imply future predictability.")
        print("  • Transaction costs and slippage not modelled.")
        print("  • All predictions are percentage next-day close moves.")
        print()

        self._print_qualitative(df)

    def _comparison_df(self) -> pd.DataFrame:
        rows = [{k: v for k, v in r.items() if k != "y_pred"} for r in self.results]
        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_csv(cfg.outputs_dir / "model_comparison.csv", index=False)
        return df

    def _save_model(self, name: str, model: object, feature_cols: List[str]) -> None:
        model_path = cfg.models_dir / f"{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_path)
        logger.success(f"  Saved → {model_path}")

        # Persist feature columns alongside the model so downstream loaders know the schema
        feat_path = cfg.models_dir / f"{name.lower().replace(' ', '_')}_feature_cols.pkl"
        joblib.dump(feature_cols, feat_path)
        logger.debug(f"  Saved feature cols ({len(feature_cols)}) → {feat_path}")

        try:
            if hasattr(model, "feature_importances_"):
                fi = pd.DataFrame({
                    "feature": feature_cols,
                    "importance": model.feature_importances_
                }).sort_values("importance", ascending=False)
                fi.to_csv(
                    cfg.outputs_dir / f"{name.lower()}_feature_importance.csv",
                    index=False,
                )
                logger.debug(f"  Top-5 ({name}): {fi.head(5)['feature'].tolist()}")
        except Exception as e:
            logger.warning(f"Could not save feature importance for {name}: {e}")

    @staticmethod
    def _print_qualitative(df: pd.DataFrame) -> None:
        print("STRENGTHS & WEAKNESSES")
        print("─" * 70)
        for _, row in df.iterrows():
            if row["model"] == "Naive Baseline":
                continue
            da = row["DirectionalAccuracy"]
            mae = row["MAE"]
            rmse = row["RMSE"]
            print(f"\n  {row['model']}:")
            if da > 0.55:
                print(f"    + Strong directional accuracy ({da:.1%}) — viable for long/short signals.")
            elif da > 0.50:
                print(f"    ~ Marginal directional accuracy ({da:.1%}) — consider ensemble.")
            else:
                print(f"    - Below-chance direction ({da:.1%}) — needs more feature engineering.")
            if rmse / mae > 1.5:
                print("    - RMSE >> MAE — occasional large errors. Check for outlier dates.")
            else:
                print("    + Error distribution is stable (RMSE ≈ MAE).")
        print()


if __name__ == "__main__":
    from src.dataset_builder import DatasetBuilder
    db = DatasetBuilder()
    X_train, X_val, X_test, y_train, y_val, y_test, feat_cols = db.build()
    mt = ModelTrainer()
    results = mt.train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, feat_cols)
    mt.print_comparison(results)
