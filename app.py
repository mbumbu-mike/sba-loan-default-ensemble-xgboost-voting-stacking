import streamlit as st
import joblib
import numpy as np

# ============================
# LOAD MODEL + FEATURES
# ============================
model = joblib.load("xgb_loan_model.pkl")
feature_names = [
    'term',
    'sba_appv',
    'disbursement_date_years_since_2015',
    'rev_line_cr_clean_Y',
    'franchise_code_1',
    'bank_bbcn bank',
    'bank_state_ca',
    'new_exist_New',
    'urban_rural_Urban',
    'bank_state_nc'
]

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Loan Default Risk Engine",
    page_icon="💳",
    layout="wide"
)

# ============================
# HEADER
# ============================
st.title("💳 Loan Default Risk Prediction System")
st.markdown("### Powered by XGBoost Machine Learning Model")

st.markdown("---")

# ============================
# LAYOUT
# ============================
col1, col2 = st.columns([2, 1])

# ============================
# INPUT PANEL
# ============================
with col1:
    st.subheader("📥 Enter Loan Application Details")

    term = st.number_input("Loan Term (months)", value=0)
    sba_appv = st.number_input("SBA Approved Amount", value=0.0)
    disbursement_date_years_since_2015 = st.number_input("Years Since 2015 (Disbursement)", value=0)
    
    rev_line_cr_clean_Y = st.selectbox("Revolving Credit Line", ["No", "Yes"])
    franchise_code_1 = st.selectbox("Franchise", ["No", "Yes"])
    bank_bbcn_bank = st.selectbox("Bank: BBCN Bank", ["No", "Yes"])
    bank_state_ca = st.selectbox("Bank State: California", ["No", "Yes"])
    new_exist_New = st.selectbox("New Business", ["No", "Yes"])
    urban_rural_Urban = st.selectbox("Location Type (Urban)", ["No", "Yes"])
    bank_state_nc = st.selectbox("Bank State: North Carolina", ["No", "Yes"])

    # Convert categorical to numeric
    input_data = np.array([[
        term,
        sba_appv,
        disbursement_date_years_since_2015,
        1 if rev_line_cr_clean_Y == "Yes" else 0,
        1 if franchise_code_1 == "Yes" else 0,
        1 if bank_bbcn_bank == "Yes" else 0,
        1 if bank_state_ca == "Yes" else 0,
        1 if new_exist_New == "Yes" else 0,
        1 if urban_rural_Urban == "Yes" else 0,
        1 if bank_state_nc == "Yes" else 0
    ]])

# ============================
# PREDICTION PANEL
# ============================
with col2:
    st.subheader("📊 Risk Score")

    if st.button("🔍 Predict Default Risk", use_container_width=True):

        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        st.metric(label="Default Probability", value=f"{prob:.2%}")

        if pred == 1:
            st.error("⚠ High Risk Loan")
        else:
            st.success("✅ Low Risk Loan")

        st.progress(float(prob))

        # Risk interpretation
        if prob < 0.3:
            st.info("Low Risk Zone")
        elif prob < 0.7:
            st.warning("Medium Risk Zone")
        else:
            st.error("High Risk Zone")

# ============================
# FOOTER
# ============================
st.markdown("---")
st.caption("Credit Risk Model | XGBoost | SBA Loan Default Prediction System")