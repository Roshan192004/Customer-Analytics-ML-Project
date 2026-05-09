import streamlit as st

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stSidebar"] .stRadio label {
    color: #c4c4e0 !important;
    font-size: 15px;
}

/* Metric Cards */
.metric-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(100,100,255,0.2);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-label {
    font-size: 0.85rem;
    color: #9ca3af;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-delta {
    font-size: 0.8rem;
    margin-top: 4px;
}

/* Section headers */
.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #c4b5fd;
    margin: 20px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(196,181,253,0.3);
}

/* Page hero */
.page-hero {
    background: linear-gradient(135deg, rgba(139,92,246,0.3), rgba(59,130,246,0.2));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 28px;
    backdrop-filter: blur(10px);
}
.page-hero h1 { margin: 0; font-size: 2rem; color: #f0eaff; }
.page-hero p  { margin: 8px 0 0; color: #c4b5fd; font-size: 1rem; }

/* Inputs */
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #c4b5fd !important;
    font-weight: 500;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100%;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Result boxes */
.result-high {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.5);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.result-low {
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.5);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.rec-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
}
.tag {
    display: inline-block;
    background: rgba(139,92,246,0.3);
    color: #c4b5fd;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 3px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Customer Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Dashboard", "🔮  Churn Prediction", "🎯  Recommendations"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7280'>Telco Customer Churn<br>DS Project · Phase 10</small>",
        unsafe_allow_html=True
    )

# ── Route Pages ───────────────────────────────────────────────
if page == "🏠  Dashboard":
    from pages import dashboard
    dashboard.show()
elif page == "🔮  Churn Prediction":
    from pages import prediction
    prediction.show()
elif page == "🎯  Recommendations":
    from pages import recommendation
    recommendation.show()
