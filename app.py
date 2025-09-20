# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

st.set_page_config(page_title="Fraud Detection • Demo", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# small CSS to tidy up look
# -------------------------
st.markdown(
    """
    <style>
        .reportview-container { font-family: "Segoe UI", Roboto, Arial; }
        .main .block-container{ padding: 1.2rem 2rem; }
        header .decoration {display:none;}
        .stDownloadButton>button { background-color:#0a84ff; color:white; }
        .big-number { font-size: 36px; font-weight:700; }
    </style>
    """, unsafe_allow_html=True
)

# -------------------------
# caching decorator compatibility
# -------------------------
try:
    cache_resource = st.cache_resource
except AttributeError:
    cache_resource = st.cache(allow_output_mutation=True)

# -------------------------
# load artifacts
# -------------------------
@cache_resource
def load_model_and_features(model_path="fraud_model.pkl", feat_path="feature_list.pkl"):
    model = joblib.load(model_path)
    feature_list = joblib.load(feat_path)
    return model, feature_list

model, FEATURE_LIST = load_model_and_features()

# -------------------------
# helper: preprocessing to match training
# -------------------------
def preprocess_input(df, feature_list):
    # Columns to drop (same as training)
    drop_cols = ['CustomerID', 'LastLogin', 'Name', 'Address', 
                 'TransactionID', 'Timestamp', 'SuspiciousFlag', 'MerchantID']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # One-hot encode 'Category' if present
    if 'Category' in df.columns:
        df = pd.get_dummies(df, columns=['Category'], drop_first=True)
    
    # Ensure every feature the model expects exists in df (add missing as 0)
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    # Align column order exactly
    df = df[feature_list]
    return df

# -------------------------
# UI layout
# -------------------------
st.title("💳 Fraud Detection • Demo")
st.caption("Upload transactions (CSV) or create a single transaction and get fraud predictions (powered by XGBoost).")

# two-column layout: left = controls, right = results
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("Operations")
    mode = st.selectbox("Mode", ["Batch predictions (CSV)", "Single transaction (manual/edit)"])

    st.write("Model info:")
    st.write(f"- Features used: {len(FEATURE_LIST)}")
    if st.button("Show top 10 model features"):
        fi = pd.Series(model.feature_importances_, index=FEATURE_LIST).sort_values(ascending=False).head(10)
        st.bar_chart(fi)

    st.markdown("---")
    st.write("Tips:")
    st.write("• If doing single prediction, edit the row values and press Predict. \n• For batch, ensure column names match original dataset (before get_dummies Category becomes Category_<val>).")

with right_col:
    if mode == "Batch predictions (CSV)":
        st.subheader("Upload a CSV file")
        uploaded = st.file_uploader("Upload CSV with transactions (<= 10k rows recommended)", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            if st.button("Run batch predictions"):
                # Preprocess
                X_input = preprocess_input(df.copy(), FEATURE_LIST)
                preds = model.predict(X_input)
                probas = model.predict_proba(X_input)[:, 1]

                results = df.copy()
                results["Predicted"] = preds
                results["Fraud_Probability"] = probas

                st.success(f"Predicted {len(results)} rows.")
                st.dataframe(results.head(100))

                # If ground truth column present, show metrics
                if 'FraudIndicator' in df.columns:
                    y_true = df['FraudIndicator'].values
                    y_pred = preds
                    y_proba = probas
                    st.markdown("### Metrics (using uploaded `FraudIndicator` column)")
                    st.text(classification_report(y_true, y_pred, digits=4))
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', 
                                xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
                    st.pyplot(fig)
                    try:
                        roc = roc_auc_score(y_true, y_proba)
                        st.write(f"ROC AUC: {roc:.4f}")
                    except Exception:
                        pass

                # Offer download
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button("Download results CSV", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

    else:
        # Single transaction mode
        st.subheader("Single transaction (create or edit values)")
        # create one-row DataFrame with default zeros
        sample_row = pd.DataFrame([ {c: 0 for c in FEATURE_LIST} ])
        edited = st.experimental_data_editor(sample_row, num_rows="fixed", use_container_width=True)

        if st.button("Predict this transaction"):
            X_single = preprocess_input(edited.copy(), FEATURE_LIST)
            pred = model.predict(X_single)[0]
            proba = model.predict_proba(X_single)[0, 1]

            # Nice result card
            col1, col2 = st.columns([1, 2])
            with col1:
                if pred == 1:
                    st.markdown("<div style='background:#ffeceb;padding:12px;border-radius:8px'>"
                                "<div class='big-number'>🚨 FRAUD</div>"
                                f"<div>Probability: <b>{proba:.4f}</b></div></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background:#e9f9ff;padding:12px;border-radius:8px'>"
                                "<div class='big-number'>✅ NOT FRAUD</div>"
                                f"<div>Probability: <b>{proba:.4f}</b></div></div>", unsafe_allow_html=True)
            with col2:
                # show top contributing features (using feature importances)
                fi = pd.Series(model.feature_importances_, index=FEATURE_LIST).sort_values(ascending=False).head(8)
                st.markdown("**Top model features**")
                st.bar_chart(fi)

            # show the input values + predicted label
            st.markdown("**Transaction details (as passed to model)**")
            display_df = edited.copy()
            display_df["Predicted"] = pred
            display_df["Fraud_Probability"] = proba
            st.dataframe(display_df.T)

# footer
st.markdown("---")
st.caption("Built with Streamlit • Keep your model files (fraud_model.pkl & feature_list.pkl) in repo root.")
