import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SERVICE_MAP = {
    'PhoneService_Yes': '📞 Phone Service',
    'MultipleLines_Yes': '📱 Multiple Lines',
    'OnlineSecurity_Yes': '🔒 Online Security',
    'OnlineBackup_Yes': '☁️ Online Backup',
    'DeviceProtection_Yes': '🛡️ Device Protection',
    'TechSupport_Yes': '🧰 Tech Support',
    'StreamingTV_Yes': '📺 Streaming TV',
    'StreamingMovies_Yes': '🎬 Streaming Movies',
}

@st.cache_resource
def build_knn():
    df = pd.read_csv('data/cleaned_data.csv')
    X = df.drop('Churn', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(X_scaled)
    return df, X, scaler, knn

def show():
    st.markdown("""
    <div class="page-hero">
        <h1>🎯 Smart Recommendations</h1>
        <p>Enter customer details to find their 5 most similar peers and discover which services to recommend</p>
    </div>
    """, unsafe_allow_html=True)

    df, X, scaler, knn = build_knn()
    feature_cols = list(X.columns)

    # ── Input Form ────────────────────────────────────────────
    st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📋 Account Info**")
        tenure = st.slider("Tenure (months)", 0, 72, 24, key="rec_tenure")
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 55, key="rec_monthly")
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0,
                                         value=float(monthly_charges * tenure), key="rec_total")
        contract_month = st.checkbox("Month-to-Month Contract", value=True, key="rec_c1")
        contract_one   = st.checkbox("One-Year Contract", key="rec_c2")
        contract_two   = st.checkbox("Two-Year Contract", key="rec_c3")
        paperless      = st.checkbox("Paperless Billing", value=True, key="rec_pb")

    with c2:
        st.markdown("**🌐 Current Services**")
        phone_service   = st.checkbox("Phone Service", value=True, key="rec_ph")
        multiple_lines  = st.checkbox("Multiple Lines", key="rec_ml")
        internet_dsl    = st.checkbox("Internet: DSL", key="rec_dsl")
        internet_fiber  = st.checkbox("Internet: Fiber Optic", value=True, key="rec_fib")
        online_security = st.checkbox("Online Security", key="rec_os")
        online_backup   = st.checkbox("Online Backup", key="rec_ob")
        device_prot     = st.checkbox("Device Protection", key="rec_dp")
        tech_support    = st.checkbox("Tech Support", key="rec_ts")
        streaming_tv    = st.checkbox("Streaming TV", key="rec_stv")
        streaming_movies = st.checkbox("Streaming Movies", key="rec_sm")

    with c3:
        st.markdown("**👤 Demographics**")
        gender_male = st.checkbox("Gender: Male", value=True, key="rec_gm")
        senior      = st.checkbox("Senior Citizen", key="rec_sc")
        partner     = st.checkbox("Has Partner", value=True, key="rec_par")
        dependents  = st.checkbox("Has Dependents", key="rec_dep")
        pay_bank    = st.checkbox("Payment: Bank Transfer", key="rec_pb2")
        pay_cc      = st.checkbox("Payment: Credit Card", key="rec_cc")
        pay_echeck  = st.checkbox("Payment: Electronic Check", value=True, key="rec_ec")
        pay_mcheck  = st.checkbox("Payment: Mailed Check", key="rec_mc")

    st.markdown("<br>", unsafe_allow_html=True)
    rec_btn = st.button("🎯 Get Recommendations")

    if rec_btn:
        # Build input vector
        input_dict = {col: 0 for col in feature_cols}
        input_dict['tenure'] = tenure
        input_dict['MonthlyCharges'] = monthly_charges
        input_dict['TotalCharges'] = total_charges

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

        # KNN lookup
        distances, indices = knn.kneighbors(input_scaled)
        neighbor_idx = indices[0][1:]  # exclude self-match
        neighbor_data = df.iloc[neighbor_idx]

        # Determine which services customer DOESN'T have
        current_services = pd.Series(input_dict)
        service_cols = list(SERVICE_MAP.keys())
        missing = [col for col in service_cols if col in input_dict and input_dict[col] == 0]

        st.markdown("---")
        col_r, col_n = st.columns([1, 1])

        with col_r:
            st.markdown('<div class="section-title">💡 Recommended Services</div>', unsafe_allow_html=True)
            if not missing:
                st.success("✅ This customer already has all services!")
            else:
                scores = neighbor_data[missing].mean().sort_values(ascending=False)
                top_recs = scores[scores > 0]
                if top_recs.empty:
                    st.info("No strong recommendations based on similar customers.")
                else:
                    for svc_col, score in top_recs.items():
                        label = SERVICE_MAP.get(svc_col, svc_col)
                        pct = int(score * 100)
                        st.markdown(f"""
                        <div class="rec-card">
                            <div style="display:flex; justify-content:space-between; align-items:center">
                                <div style="font-size:1rem; color:#e8e8f0; font-weight:500">{label}</div>
                                <span class="tag">{pct}% of peers use it</span>
                            </div>
                            <div style="margin-top:8px; background:rgba(255,255,255,0.05); border-radius:8px; height:6px">
                                <div style="background:linear-gradient(90deg,#7c3aed,#a78bfa); width:{pct}%; height:6px; border-radius:8px"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        with col_n:
            st.markdown('<div class="section-title">👥 5 Most Similar Customers</div>', unsafe_allow_html=True)
            display_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
            available = [c for c in display_cols if c in neighbor_data.columns]
            peer_df = neighbor_data[available].copy().reset_index(drop=True)
            peer_df.index = peer_df.index + 1
            peer_df.columns = ['Tenure (mo)', 'Monthly ($)', 'Total ($)', 'Churned']
            peer_df['Churned'] = peer_df['Churned'].map({0: '✅ No', 1: '⚠️ Yes'})
            st.dataframe(peer_df, use_container_width=True)

            # Peer similarity chart
            sim_scores = [round((1 - d) * 100, 1) for d in distances[0][1:]]
            fig = px.bar(
                x=[f"Peer {i+1}" for i in range(len(sim_scores))],
                y=sim_scores,
                color=sim_scores,
                color_continuous_scale=['#4f46e5', '#a78bfa'],
                labels={'x': 'Similar Customer', 'y': 'Similarity (%)'},
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#c4b5fd', coloraxis_showscale=False,
                title='Similarity Scores', title_font_color='#c4b5fd',
                margin=dict(t=30, b=10, l=10, r=10)
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
            st.plotly_chart(fig, use_container_width=True)
