import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def train_model():
    df = pd.read_csv('data/cleaned_data.csv')
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(
        n_estimators=50, max_depth=10, min_samples_split=5, random_state=42
    )
    model.fit(X_scaled, y)
    return model, scaler, list(X.columns)

def show():
    st.markdown("""
    <div class="page-hero">
        <h1>🔮 Churn Prediction</h1>
        <p>Enter customer details to predict their likelihood of churning using the optimized Random Forest model</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading optimized model..."):
        model, scaler, feature_cols = train_model()

    # ── Two-Column Form ───────────────────────────────────────
    st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Account Info**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(monthly_charges * tenure))

        contract_month = st.checkbox("Month-to-Month Contract", value=True)
        contract_one  = st.checkbox("One-Year Contract")
        contract_two  = st.checkbox("Two-Year Contract")
        paperless     = st.checkbox("Paperless Billing", value=True)

    with col2:
        st.markdown("**🌐 Internet & Services**")
        internet_dsl   = st.checkbox("Internet: DSL")
        internet_fiber = st.checkbox("Internet: Fiber Optic", value=True)
        online_security = st.checkbox("Online Security")
        online_backup   = st.checkbox("Online Backup")
        device_prot     = st.checkbox("Device Protection")
        tech_support    = st.checkbox("Tech Support")
        streaming_tv    = st.checkbox("Streaming TV")
        streaming_movies = st.checkbox("Streaming Movies")

    with col3:
        st.markdown("**📞 Phone & Payment**")
        phone_service = st.checkbox("Phone Service", value=True)
        multiple_lines = st.checkbox("Multiple Lines")
        pay_bank       = st.checkbox("Payment: Bank Transfer")
        pay_cc         = st.checkbox("Payment: Credit Card")
        pay_echeck     = st.checkbox("Payment: Electronic Check", value=True)
        pay_mcheck     = st.checkbox("Payment: Mailed Check")
        gender_male    = st.checkbox("Gender: Male", value=True)
        senior         = st.checkbox("Senior Citizen")
        partner        = st.checkbox("Has Partner", value=True)
        dependents     = st.checkbox("Has Dependents")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Churn Risk")

    if predict_btn:
        # Build feature vector matching training columns
        input_dict = {col: 0 for col in feature_cols}

        # Numeric
        input_dict['tenure'] = tenure
        input_dict['MonthlyCharges'] = monthly_charges
        input_dict['TotalCharges'] = total_charges

        # Boolean mappings
        bool_map = {
            'SeniorCitizen': senior,
            'Partner_Yes': partner,
            'Dependents_Yes': dependents,
            'PhoneService_Yes': phone_service,
            'MultipleLines_Yes': multiple_lines,
            'InternetService_DSL': internet_dsl,
            'InternetService_Fiber optic': internet_fiber,
            'OnlineSecurity_Yes': online_security,
            'OnlineBackup_Yes': online_backup,
            'DeviceProtection_Yes': device_prot,
            'TechSupport_Yes': tech_support,
            'StreamingTV_Yes': streaming_tv,
            'StreamingMovies_Yes': streaming_movies,
            'Contract_Month-to-month': contract_month,
            'Contract_One year': contract_one,
            'Contract_Two year': contract_two,
            'PaperlessBilling_Yes': paperless,
            'PaymentMethod_Bank transfer (automatic)': pay_bank,
            'PaymentMethod_Credit card (automatic)': pay_cc,
            'PaymentMethod_Electronic check': pay_echeck,
            'PaymentMethod_Mailed check': pay_mcheck,
            'gender_Male': gender_male,
        }
        for key, val in bool_map.items():
            if key in input_dict:
                input_dict[key] = int(val)

        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]

        st.markdown("---")
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            if pred == 1:
                st.markdown(f"""
                <div class="result-high">
                    <div style="font-size:3rem">⚠️</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#f87171; margin:10px 0">HIGH CHURN RISK</div>
                    <div style="color:#fca5a5; font-size:1rem">This customer has a <b>{prob*100:.1f}%</b> probability of churning.</div>
                    <div style="margin-top:14px; color:#9ca3af; font-size:0.85rem">Consider retention incentives, loyalty discounts, or proactive outreach.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <div style="font-size:3rem">✅</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#34d399; margin:10px 0">LOW CHURN RISK</div>
                    <div style="color:#6ee7b7; font-size:1rem">This customer has only a <b>{prob*100:.1f}%</b> probability of churning.</div>
                    <div style="margin-top:14px; color:#9ca3af; font-size:0.85rem">Customer appears stable. Focus on upselling or rewards.</div>
                </div>
                """, unsafe_allow_html=True)

        with gauge_col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={'suffix': "%", 'font': {'color': '#c4b5fd', 'size': 40}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#9ca3af'},
                    'bar': {'color': '#f87171' if pred == 1 else '#34d399'},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 40],  'color': 'rgba(52,211,153,0.15)'},
                        {'range': [40, 70], 'color': 'rgba(251,191,36,0.15)'},
                        {'range': [70, 100],'color': 'rgba(248,113,113,0.15)'},
                    ],
                    'threshold': {
                        'line': {'color': '#fbbf24', 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                },
                title={'text': 'Churn Probability', 'font': {'color': '#c4b5fd'}}
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#c4b5fd'},
                height=280,
                margin=dict(t=30, b=10, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
