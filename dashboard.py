import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# Configure page layout
st.set_page_config(
    page_title="Enterprise AI Telemetry | Risk & Drift Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Dark-Mode CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #F1F5F9;
    }

    .stApp {
        background-color: #07090E;
    }

    /* Enterprise Headers */
    .console-header {
        border-bottom: 1px solid #1E293B;
        padding-bottom: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .console-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.4rem;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        text-transform: uppercase;
        margin: 0;
    }
    .console-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.25rem;
        letter-spacing: 0.05em;
    }

    /* Metric Cards */
    .metric-panel {
        background: #0D131F;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 1.1rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
    }
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 0.3rem;
    }

    /* Status Badges */
    .badge-container {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-stable {
        background-color: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .badge-risk {
        background-color: rgba(239, 68, 68, 0.12);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-stable { background-color: #10B981; }
    .dot-risk { background-color: #EF4444; }

    /* Button Customization */
    div.stButton > button {
        border-radius: 4px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# Top Telemetry Header
st.markdown("""
<div class="console-header">
    <div class="console-title">Enterprise Risk Engine & Continuous Drift Monitor</div>
    <div class="console-subtitle">SYSTEM ARCHITECTURE: PYTORCH DEEP LEARNING CORE // FASTAPI ASYNC INFERENCE // SQL AUDIT BUS</div>
</div>
""", unsafe_allow_html=True)

API_BASE = "http://127.0.0.1:8000"

tab_inference, tab_telemetry = st.tabs(["01 // REAL-TIME INFERENCE", "02 // PRODUCTION MONITORING & DRIFT"])

# ==========================================
# TAB 1: REAL-TIME INFERENCE ENGINE
# ==========================================
with tab_inference:
    col_inputs, col_outputs = st.columns([1, 1], gap="large")

    with col_inputs:
        st.markdown('<div class="metric-label" style="margin-bottom: 0.75rem;">INPUT FEATURE VECTOR</div>', unsafe_allow_html=True)
        
        usage = st.slider("Monthly System Usage (Hours)", min_value=5.0, max_value=150.0, value=38.0, step=1.0)
        tickets = st.slider("Support Incident Tickets Raised", min_value=0, max_value=10, value=3, step=1)
        contract = st.select_slider("Contract Term Commitment (Months)", options=[1, 6, 12, 24], value=6)
        bill = st.number_input("Monthly Billed Invoice Amount (USD)", min_value=30.0, max_value=500.0, value=115.0, step=5.0)

        predict_btn = st.button("RUN PYTORCH INFERENCE", use_container_width=True, type="primary")

    with col_outputs:
        st.markdown('<div class="metric-label" style="margin-bottom: 0.75rem;">INFERENCE TELEMETRY & DECISION MATRIX</div>', unsafe_allow_html=True)

        if predict_btn:
            payload = {
                "usage_hours": float(usage),
                "support_tickets": int(tickets),
                "contract_months": int(contract),
                "monthly_bill": float(bill)
            }
            try:
                response = requests.post(f"{API_BASE}/predict", json=payload, timeout=5)
                data = response.json()
                
                score = data["churn_risk_score"]
                prediction = data["prediction"]
                latency = data["latency_ms"]

                # Metrics Row
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.markdown(f"""
                    <div class="metric-panel">
                        <div class="metric-label">Inference Latency</div>
                        <div class="metric-value">{latency} <span style="font-size: 0.9rem; color: #64748B;">ms</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                with m_col2:
                    is_risk = score > 0.5
                    badge_class = "badge-risk" if is_risk else "badge-stable"
                    dot_class = "dot-risk" if is_risk else "dot-stable"
                    status_text = "HIGH RISK PROFILE" if is_risk else "ACCOUNT STABLE"
                    
                    st.markdown(f"""
                    <div class="metric-panel">
                        <div class="metric-label">Model Classification</div>
                        <div style="margin-top: 0.4rem;">
                            <span class="badge-container {badge_class}">
                                <span class="status-dot {dot_class}"></span>
                                {status_text}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Modern Plotly Gauge Chart for Risk Probability
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score * 100,
                    number={'suffix': "%", 'font': {'color': "#F8FAFC", 'size': 36, 'family': 'Inter'}},
                    title={'text': "CALIBRATED RISK PROBABILITY", 'font': {'size': 11, 'color': "#94A3B8", 'family': 'JetBrains Mono'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
                        'bar': {'color': "#EF4444" if score > 0.5 else "#3B82F6"},
                        'bgcolor': "#0B0F19",
                        'borderwidth': 1,
                        'bordercolor': "#1E293B",
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(59, 130, 246, 0.08)"},
                            {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.08)"}
                        ],
                        'threshold': {
                            'line': {'color': "#EF4444", 'width': 2},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(
                    paper_bgcolor="#07090E",
                    plot_bgcolor="#07090E",
                    height=200,
                    margin=dict(l=20, r=20, t=30, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error("Connection Refused: Confirm the FastAPI service is active on 127.0.0.1:8000.")
        else:
            st.markdown("""
            <div style="border: 1px dashed #1E293B; border-radius: 6px; padding: 3rem 1.5rem; text-align: center;">
                <div class="metric-label" style="color: #475569;">Awaiting Input Execution</div>
                <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.3rem;">Modify feature parameters on the left and trigger inference execution.</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 2: MONITORING & DRIFT TELEMETRY
# ==========================================
with tab_telemetry:
    col_drift_action, col_sim_action = st.columns([1, 1])
    
    with col_drift_action:
        refresh_audit = st.button("EXECUTE STATISTICAL DRIFT AUDIT", use_container_width=True)
    with col_sim_action:
        sim_traffic = st.button("SIMULATE BATCH TRAFFIC (15 REQUESTS)", use_container_width=True)

    if sim_traffic:
        try:
            r = requests.post(f"{API_BASE}/simulate-batch", timeout=5).json()
            st.success(f"Audit log updated: {r['message']}")
        except Exception:
            st.error("Batch simulation failed: FastAPI backend offline.")

    try:
        drift_data = requests.get(f"{API_BASE}/drift-check", timeout=5).json()
        logs_data = requests.get(f"{API_BASE}/logs", timeout=5).json()

        # Top Metric Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label">Inference Logged</div>
                <div class="metric-value">{drift_data['samples_analyzed']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi2:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label">Active Mean Risk</div>
                <div class="metric-value">{drift_data['current_mean_risk'] * 100:.1f}<span style="font-size: 0.9rem; color: #64748B;">%</span></div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label">Baseline Target</div>
                <div class="metric-value">{drift_data['baseline_mean_risk'] * 100:.1f}<span style="font-size: 0.9rem; color: #64748B;">%</span></div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            has_drift = drift_data['drift_detected']
            badge_cls = "badge-risk" if has_drift else "badge-stable"
            dot_cls = "dot-risk" if has_drift else "dot-stable"
            status_lbl = "DRIFT DETECTED" if has_drift else "STABLE DISTRIBUTION"

            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label">Drift Status</div>
                <div style="margin-top: 0.4rem;">
                    <span class="badge-container {badge_cls}">
                        <span class="status-dot {dot_cls}"></span>
                        {status_lbl}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Plotly Area Chart for Score Distribution
        if logs_data:
            df = pd.DataFrame(logs_data)
            df = df.iloc[::-1].reset_index(drop=True)  # chronological order
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=df.index,
                y=df['score'],
                mode='lines+markers',
                name='Confidence Score',
                line=dict(color='#3B82F6', width=2),
                marker=dict(size=5, color='#60A5FA'),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.06)'
            ))
            
            # Baseline reference line
            fig_trend.add_hline(
                y=drift_data['baseline_mean_risk'], 
                line_dash="dot", 
                line_color="#94A3B8", 
                annotation_text="Baseline (0.38)", 
                annotation_font=dict(color="#94A3B8", size=10, family="JetBrains Mono"),
                annotation_position="bottom right"
            )

            fig_trend.update_layout(
                title={'text': "REAL-TIME SCORE DRIFT TRAJECTORY", 'font': {'size': 11, 'color': "#94A3B8", 'family': 'JetBrains Mono'}},
                paper_bgcolor="#07090E",
                plot_bgcolor="#07090E",
                height=260,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=False, color="#475569", title="Sequential Request ID"),
                yaxis=dict(showgrid=True, gridcolor="#1E293B", color="#475569", range=[0, 1])
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # SQLite Audit Trail Table
            with st.expander("INSPECT SQL AUDIT LOGS (RAW INFERENCE RECORDS)"):
                st.dataframe(
                    df[["id", "time", "usage", "tickets", "bill", "score", "prediction", "latency"]],
                    use_container_width=True,
                    hide_index=True
                )
    except Exception:
        st.warning("Backend API telemetry is currently offline.")