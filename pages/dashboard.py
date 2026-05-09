import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data():
    df = pd.read_csv('data/cleaned_data.csv')
    return df

def show():
    df = load_data()

    # ── Hero ──────────────────────────────────────────────────
    st.markdown("""
    <div class="page-hero">
        <h1>📊 Customer Analytics Dashboard</h1>
        <p>Live overview of churn trends, customer segments, and spending patterns</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Metrics ───────────────────────────────────────────
    total_customers = len(df)
    churn_rate = df['Churn'].mean() * 100
    avg_monthly = df['MonthlyCharges'].mean()
    avg_tenure = df['tenure'].mean()

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "👥", f"{total_customers:,}", "Total Customers", "#a78bfa"),
        (c2, "⚠️", f"{churn_rate:.1f}%", "Churn Rate", "#f87171"),
        (c3, "💵", f"${avg_monthly:.0f}", "Avg Monthly Charges", "#34d399"),
        (c4, "📅", f"{avg_tenure:.0f} mo", "Avg Tenure", "#60a5fa"),
    ]
    for col, icon, val, label, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem">{icon}</div>
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Churn Distribution + Charges Distribution ─────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Churn Distribution</div>', unsafe_allow_html=True)
        churn_counts = df['Churn'].value_counts().reset_index()
        churn_counts.columns = ['Churn', 'Count']
        churn_counts['Label'] = churn_counts['Churn'].map({0: 'Retained', 1: 'Churned'})
        fig = px.pie(
            churn_counts, values='Count', names='Label',
            color_discrete_sequence=['#7c3aed', '#f87171'],
            hole=0.55
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#c4b5fd', showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        fig.update_traces(textfont_color='white')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Monthly Charges Distribution</div>', unsafe_allow_html=True)
        fig2 = px.histogram(
            df, x='MonthlyCharges', color='Churn',
            color_discrete_map={0: '#7c3aed', 1: '#f87171'},
            barmode='overlay', nbins=40, opacity=0.8,
            labels={'Churn': 'Churned'}
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#c4b5fd', xaxis_title='Monthly Charges ($)',
            yaxis_title='Count', showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        fig2.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig2.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Tenure vs Churn + Contract type ────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Tenure vs Churn Rate</div>', unsafe_allow_html=True)
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,36,48,60,72], labels=['0-12','12-24','24-36','36-48','48-60','60-72'])
        tenure_churn = df.groupby('tenure_group', observed=True)['Churn'].mean().reset_index()
        tenure_churn.columns = ['Tenure (months)', 'Churn Rate']
        fig3 = px.bar(
            tenure_churn, x='Tenure (months)', y='Churn Rate',
            color='Churn Rate',
            color_continuous_scale=['#7c3aed', '#f87171'],
        )
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#c4b5fd', coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        fig3.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig3.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">Top Feature Importances</div>', unsafe_allow_html=True)
        feature_data = {
            'Feature': ['tenure', 'TotalCharges', 'MonthlyCharges', 'Fiber optic', 'Two-year Contract', 'Electronic Check', 'Online Security', 'One-year Contract', 'Tech Support', 'Paperless Billing'],
            'Importance': [0.212, 0.156, 0.119, 0.080, 0.058, 0.054, 0.036, 0.031, 0.021, 0.018]
        }
        fi_df = pd.DataFrame(feature_data).sort_values('Importance')
        fig4 = px.bar(
            fi_df, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale=['#4f46e5', '#a78bfa']
        )
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#c4b5fd', coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        fig4.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig4.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig4, use_container_width=True)

    # ── Model Performance Table ───────────────────────────────
    st.markdown('<div class="section-title">📋 Model Performance Comparison</div>', unsafe_allow_html=True)
    try:
        comp_df = pd.read_csv('data/model_comparison_results.csv')
        comp_df_styled = comp_df.style\
            .format({col: '{:.4f}' for col in comp_df.columns if col != 'Model'})\
            .background_gradient(cmap='Purples', subset=[c for c in comp_df.columns if c != 'Model'])
        st.dataframe(comp_df_styled, use_container_width=True)
    except Exception:
        st.info("Run `neural_networks.py` first to generate comparison results.")
