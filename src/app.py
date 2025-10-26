# app.py
from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tsunami Risk Decision Support", layout="wide")

# ----- Optional password gate (safe if secrets.toml is missing) -----
try:
    PASSWORD = st.secrets["APP_PASSWORD"] 
except Exception:
    PASSWORD = os.getenv("APP_PASSWORD")  

if PASSWORD:
    pw = st.text_input("Enter password", type="password")
    if pw != PASSWORD:
        st.stop()

# ----- Resolve artifacts/reports whether app is in root or in src/ -----
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
REP = HERE / "reports"
if not ART.exists() and (HERE.parent / "artifacts").exists():
    ART = HERE.parent / "artifacts"
    REP = HERE.parent / "reports"

st.title("Tsunami Risk Decision Support (TRDS)")
st.caption("This product supplements official NOAA/USGS alerts; it does not replace them.")

# ----- Check required artifacts -----
model_path = ART / "model.pkl"
metrics_path = ART / "metrics.json"
feat_path = ART / "feature_names.json"

missing = [p.name for p in (model_path, metrics_path, feat_path) if not p.exists()]
if missing:
    st.warning(
        "Required artifacts not found: " + ", ".join(missing) +
        "\n\nFrom the project root, run:\n"
        "1) `python -m src.make_dummy_data`  (or use real data prep)\n"
        "2) `python -m src.train --clean data/clean/earthquakes_clean.csv --artifacts artifacts --reports reports`\n"
        "Then re-run: `streamlit run app.py`"
    )
    st.stop()

# ----- Load artifacts -----
model = joblib.load(model_path)
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)
with open(feat_path, "r", encoding="utf-8") as f:
    feature_names = json.load(f)

threshold = float(metrics.get("threshold", 0.5))
pr_auc = float(metrics.get("pr_auc", 0.0))
roc_auc = float(metrics.get("roc_auc", 0.0))

# ----- Top metrics -----
m1, m2, m3 = st.columns(3)
m1.metric("Operating threshold", f"{threshold:.2f}")
m2.metric("PR-AUC (test)", f"{pr_auc:.3f}")
m3.metric("ROC-AUC (test)", f"{roc_auc:.3f}")

with st.expander("Classification report (test)"):
    st.json(metrics.get("class_report", {}))

# ----- Interactive what-if prediction -----
st.subheader("Interactive what-if prediction")
st.write("Enter feature values. Defaults are 0; the pipeline handles scaling.")

# Build inputs UI
inputs = {}
for f in feature_names:
    # Keep a consistent order and sane numeric default
    inputs[f] = st.number_input(f, value=0.0, format="%.6f")

X_one = pd.DataFrame([inputs], columns=feature_names)
prob = float(model.predict_proba(X_one)[0, 1])

# Optional: allow ad-hoc threshold tweak for exploration
th_col1, th_col2 = st.columns([1, 3])
with th_col1:
    custom_thresh = st.slider("Threshold (explore)", min_value=0.0, max_value=1.0, value=threshold, step=0.01)
with th_col2:
    st.metric("Predicted tsunami probability", f"{prob:.2%}")

# Action mapping
if prob >= custom_thresh:
    st.success("Action: Notify on-call and stage resources (probability ≥ threshold).")
else:
    st.info("Action: Monitor only (probability < threshold).")

# ----- Visuals -----
st.subheader("Model visuals")

cols = st.columns(3)
cm_p = REP / "confusion_matrix.png"
pr_p = REP / "pr_curve.png"
fi_p = REP / "feature_importance.png"
roc_p = REP / "roc_curve.png"
pca_p = REP / "pca_scatter.png"

if cm_p.exists():
    cols[0].image(str(cm_p), caption="Confusion Matrix")
if pr_p.exists():
    cols[1].image(str(pr_p), caption="Precision–Recall Curve")
if fi_p.exists():
    cols[2].image(str(fi_p), caption="Permutation Feature Importance")

if roc_p.exists():
    st.image(str(roc_p), caption="ROC Curve")
if pca_p.exists():
    st.image(str(pca_p), caption="PCA Scatter (descriptive)")

st.caption(
    "Model version: "
    + Path(ART / 'version.txt').read_text(encoding='utf-8').strip()
      if (ART / 'version.txt').exists() else "Model version: unknown"
)
