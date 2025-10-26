
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance

from src.utils import find_target, train_val_test_split, save_json


# ---------- plotting helpers ----------
def plot_pr(y_true, y_prob, out_path: Path) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return float(ap)


def plot_roc(y_true, y_prob, out_path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return float(auc)


def plot_confusion(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_perm_importance(model, X_df: pd.DataFrame, y_ser: pd.Series, out_path: Path, topn: int = 12):
    """
    Permutation importance on the fitted pipeline `model`.
    This permutes columns in the original feature space (X_df columns).
    """
    r = permutation_importance(model, X_df, y_ser, n_repeats=10, random_state=42, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X_df.columns).sort_values(ascending=True).tail(topn)

    plt.figure(figsize=(7, 5))
    imp.plot(kind="barh")
    plt.xlabel("Permutation importance (mean decrease)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

    return imp.sort_values(ascending=False).to_dict()


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=str, default="data/clean/earthquakes_clean.csv")
    parser.add_argument("--artifacts", type=str, default="artifacts")
    parser.add_argument("--reports", type=str, default="reports")
    args = parser.parse_args()

    clean_path = Path(args.clean)
    if not clean_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {clean_path}. Run data prep first.")

    df = pd.read_csv(clean_path)
    target = find_target(df)
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # ensure numeric only (prep should handle imputation/scale; keep non-numeric out)
    X = X.select_dtypes(include=["number"]).copy()
    if X.empty:
        raise ValueError("No numeric features found after cleaning.")

    # train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    num_cols = X_train.columns.tolist()

    # preprocess (median impute + scale)
    num_proc = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", SKStandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", num_proc, num_cols)],
        remainder="drop"
    )

    # candidate models
    logit = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    gbt = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", GradientBoostingClassifier()),
    ])

    models = {"logistic": logit, "gbt": gbt}

    # cross-validated AP on training for model selection
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, pipe in models.items():
        y_prob_cv = cross_val_predict(pipe, X_train, y_train, cv=skf, method="predict_proba")[:, 1]
        ap = average_precision_score(y_train, y_prob_cv)
        scores[name] = float(ap)
        print(f"{name} CV-AP: {ap:.4f}")

    best_name = max(scores, key=scores.get)
    best_model = models[best_name]
    print(f"Selected model: {best_name}")

    # fit on train
    best_model.fit(X_train, y_train)

    # pick threshold on validation: maximize F1 with recall priority
    y_val_prob = best_model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)
    # compute F1 only where thresholds are defined (exclude the last PR point)
    f1_vals = []
    for p, r in zip(precision[:-1], recall[:-1]):
        f1_vals.append(0.0 if (p + r) == 0 else (2 * p * r / (p + r)))
    best_idx = int(np.argmax(f1_vals)) if f1_vals else 0
    chosen_thresh = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    print(f"Chosen threshold (validation, F1-priority): {chosen_thresh:.3f}")

    # retrain on train+val, then evaluate on test
    X_tr = pd.concat([X_train, X_val], axis=0)
    y_tr = pd.concat([y_train, y_val], axis=0)
    best_model.fit(X_tr, y_tr)

    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= chosen_thresh).astype(int)

    reports_dir = Path(args.reports)
    artifacts_dir = Path(args.artifacts)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # plots + metrics
    pr_auc = plot_pr(y_test, y_test_prob, reports_dir / "pr_curve.png")
    roc_auc = plot_roc(y_test, y_test_prob, reports_dir / "roc_curve.png")
    plot_confusion(y_test, y_test_pred, reports_dir / "confusion_matrix.png")
    fi_dict = plot_perm_importance(best_model, X_test, y_test, reports_dir / "feature_importance.png")

    # descriptive PCA plot using the pipeline preprocessor (no re-imports/shadowing)
    try:
        from sklearn.decomposition import PCA

        # transform numeric test set with the fitted preprocessor
        X_test_scaled = best_model.named_steps["prep"].transform(X_test)
        pc = PCA(n_components=2, random_state=42).fit_transform(X_test_scaled)

        idx = np.random.RandomState(42).choice(len(pc), size=min(2000, len(pc)), replace=False)
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(pc[idx, 0], pc[idx, 1], c=y_test.iloc[idx], s=8, cmap="coolwarm", alpha=0.6)
        plt.title("PCA (first 2 components) — test set")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(sc, label="tsunami")
        plt.tight_layout()
        plt.savefig(reports_dir / "pca_scatter.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"PCA plot skipped: {e}")

    # save artifacts
    joblib.dump(best_model, artifacts_dir / "model.pkl")
    joblib.dump(best_model.named_steps["prep"], artifacts_dir / "preprocessor.pkl")
    with open(artifacts_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(num_cols, f, indent=2)

    class_rep = classification_report(y_test, y_test_pred, output_dict=True)
    metrics_payload = {
        "model": best_name,
        "threshold": chosen_thresh,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "class_report": class_rep,
        "cv_ap": scores,
        "feature_importance": fi_dict,
    }
    save_json(metrics_payload, artifacts_dir / "metrics.json")

    with open(artifacts_dir / "version.txt", "w", encoding="utf-8") as f:
        f.write("v1\n")

    print("Artifacts saved to artifacts/, plots to reports/ — done.")


if __name__ == "__main__":
    main()
