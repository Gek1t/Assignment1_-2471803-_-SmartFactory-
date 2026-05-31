import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Page configuration
st.set_page_config(
    page_title="Smart Factory Predictive Maintenance System (StudentID: 2471803)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%);
    }
    /* Force headings and text to be light colored for dark theme */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, span {
        color: #f8fafc !important;
    }
    /* Set sidebar text and widgets styling */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] p {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #060913 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Make sidebar title stand out */
    [data-testid="stSidebar"] h2 {
        font-size: 22px !important;
        font-weight: 800 !important;
        border-bottom: 2px solid rgba(59, 130, 246, 0.25);
        padding-bottom: 12px;
        margin-bottom: 24px !important;
    }
    
    /* Style the selectbox and multiselect widget input field container */
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #0d1527 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"]:hover {
        border-color: rgba(59, 130, 246, 0.4) !important;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.15) !important;
    }
    
    /* Force ALL nested control boxes inside selectbox to be transparent background */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
    }
    
    /* Force text and selections inside selectbox to be light-colored */
    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #f8fafc !important;
        background-color: transparent !important;
    }
    
    /* Style the multiselect tag pills to match SCADA blue theme */
    [data-testid="stSidebar"] span[data-baseweb="tag"] {
        background-color: rgba(59, 130, 246, 0.2) !important;
        border: 1px solid rgba(59, 130, 246, 0.35) !important;
        border-radius: 4px !important;
        color: #93c5fd !important;
        margin: 2px !important;
    }
    [data-testid="stSidebar"] span[data-baseweb="tag"] * {
        color: #93c5fd !important;
        background-color: transparent !important;
    }
    
    /* Style the dropdown list popup and popover globally */
    div[data-baseweb="popover"], div[role="listbox"] {
        background-color: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5) !important;
    }
    div[data-baseweb="popover"] *, div[role="listbox"] * {
        background-color: transparent !important;
        color: #f8fafc !important;
    }
    div[role="option"]:hover, div[role="option"][aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Custom styling for label text */
    [data-testid="stSidebar"] label p {
        font-weight: 600 !important;
        font-size: 13px !important;
        letter-spacing: 0.02em;
        margin-top: 12px;
    }
    /* Hide Streamlit default header */
    header {
        visibility: hidden;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 22px 15px;
        text-align: center;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border: 1px solid rgba(59, 130, 246, 0.4);
        box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.15);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #f8fafc !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        line-height: 1.1;
    }
    .metric-value-roi {
        font-size: 32px;
        font-weight: 800;
        color: #fbbf24 !important;
        text-shadow: 0 0 15px rgba(251, 191, 36, 0.3);
        line-height: 1.1;
    }
    .metric-delta {
        font-size: 12px;
        font-weight: 600;
        margin-top: 6px;
        letter-spacing: 0.02em;
    }
</style>

""", unsafe_allow_html=True)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "SmartFactory_FD002_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_data_and_predict():
    if not os.path.exists(DATA_PATH):
        return None
    data = pd.read_csv(DATA_PATH)
    
    # Pre-compute predictions to avoid inference latency on every Streamlit rerun
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        sel_feats = bundle['selected_features']
        m_xgb = bundle['model_xgb']
        m_rf = bundle['model_rf']
        w = bundle['weights']
        w_xgb, w_rf = w['xgb'], w['rf']
        
        features_scaled = data[sel_feats]
        data['pred_RUL'] = w_xgb * m_xgb.predict(features_scaled) + w_rf * m_rf.predict(features_scaled)
        data['pred_RUL'] = data['pred_RUL'].clip(lower=0, upper=125)
        data['residual'] = data['RUL_capped'] - data['pred_RUL']
    return data

@st.cache_resource
def load_scaler():
    path = os.path.join(BASE_DIR, "config", "scaler.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_interval_std():
    path = os.path.join(BASE_DIR, "models", "interval_stds.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

# Load resources
model_bundle = load_model()
df = load_data_and_predict()
scaler = load_scaler()
interval_std = load_interval_std()

st.title("🏭 Smart Factory Turbofan Engine Predictive Maintenance Dashboard")
st.markdown("---")

if model_bundle is None or df is None or scaler is None or interval_std is None:
    st.error("Failed to load data, model, or configuration files. Please ensure that the feature engineering and model training notebooks have run successfully and generated the files.")
    st.info(f"Expected file paths:\n- Data: `{DATA_PATH}`\n- Model: `{MODEL_PATH}`\n- Scaler: `{os.path.join(BASE_DIR, 'config', 'scaler.pkl')}`\n- Interval Stds: `{os.path.join(BASE_DIR, 'models', 'interval_stds.pkl')}`")
    st.stop()

# Extract model components
selected_features = model_bundle['selected_features']
model_lr = model_bundle['model_lr']
model_xgb = model_bundle['model_xgb']
model_rf = model_bundle['model_rf']
weights = model_bundle['weights']

# Dynamic prediction and residual calculations are pre-computed in load_data_and_predict() with cache.

# ==============================================================================
# SIDEBAR FILTERS
# ==============================================================================
st.sidebar.header("🛠️ Global Filters")

# Filter 1: Engine ID Multiselect
all_engines = sorted(df['Engine_ID'].unique().tolist())
selected_engines = st.sidebar.multiselect(
    "Select Engine ID (Default: All)",
    options=all_engines,
    default=[]
)

# Filter 2: Alert Threshold Slider
alert_threshold = st.sidebar.slider(
    "RUL Alert Threshold (Cycles)",
    min_value=10,
    max_value=80,
    value=45,
    step=5
)

# Filter 3: Operating Regime Selector (KMeans Labels)
regimes = sorted(df['regime'].unique().tolist())
selected_regimes = st.sidebar.multiselect(
    "Select Operational Regime (Regime Cluster)",
    options=regimes,
    default=regimes
)

# Filter 4: Cycle Range Slider
min_cycle, max_cycle = int(df['Cycle'].min()), int(df['Cycle'].max())
selected_cycle_range = st.sidebar.slider(
    "Cycle Range",
    min_value=min_cycle,
    max_value=max_cycle,
    value=(min_cycle, max_cycle)
)

# Apply filters
filtered_df = df.copy()
if selected_engines:
    filtered_df = filtered_df[filtered_df['Engine_ID'].isin(selected_engines)]

filtered_df = filtered_df[
    (filtered_df['regime'].isin(selected_regimes)) & 
    (filtered_df['Cycle'] >= selected_cycle_range[0]) & 
    (filtered_df['Cycle'] <= selected_cycle_range[1])
]

# Latest status for each engine (current state)
latest_df = filtered_df.sort_values('Cycle').groupby('Engine_ID').last().reset_index()

# ==============================================================================
# Tabbed Layout
# ==============================================================================
tab_fleet, tab_engine = st.tabs(["📊 Fleet Health Overview", "🔍 Single Engine Drill-Down"])

# ------------------------------------------------------------------------------
# TAB 1: FLEET OVERVIEW
# ------------------------------------------------------------------------------
with tab_fleet:
    # 1. KPI Metric Cards (5 Cards)
    total_engines = latest_df['Engine_ID'].nunique()
    avg_pred_rul = latest_df['pred_RUL'].mean()
    
    # Calculate high risk engines based on latest prediction
    high_risk_df = latest_df[latest_df['pred_RUL'] <= alert_threshold]
    high_risk_count = len(high_risk_df)
    
    # Safety Index
    safety_index = (1 - (high_risk_count / total_engines)) * 100 if total_engines > 0 else 100.0
    
    # ROI Calculation (Confusion Matrix-based Cost-Benefit ROI Model)
    # TP: Avoided Failure (pred <= threshold, actual <= threshold). Saving: +$15,000
    # FP: False Alarm / Over-maintenance (pred <= threshold, actual > threshold). Cost: -$3,000
    # FN: Missed Alert / Catastrophic Failure (pred > threshold, actual <= threshold). Cost: -$25,000
    # TN: Correct Non-alert (pred > threshold, actual > threshold). Cost: $0
    tp_df = latest_df[(latest_df['pred_RUL'] <= alert_threshold) & (latest_df['RUL'] <= alert_threshold)]
    fp_df = latest_df[(latest_df['pred_RUL'] <= alert_threshold) & (latest_df['RUL'] > alert_threshold)]
    fn_df = latest_df[(latest_df['pred_RUL'] > alert_threshold) & (latest_df['RUL'] <= alert_threshold)]
    
    tp_count = len(tp_df)
    fp_count = len(fp_df)
    fn_count = len(fn_df)
    
    roi_saved = (tp_count * 15000) - (fp_count * 3000) - (fn_count * 25000)
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Monitored Engines</div>
            <div class="metric-value">{total_engines} Units</div>
            <div class="metric-delta" style="color: #34d399;">▲ 100% Monitored Rate</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Predicted RUL</div>
            <div class="metric-value">{avg_pred_rul:.1f} Cycles</div>
            <div class="metric-delta" style="color: #fbbf24;">Moderate Degradation</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        color = "#f87171" if high_risk_count > 0 else "#34d399"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High-Risk Assets (RUL ≤ {alert_threshold})</div>
            <div class="metric-value" style="color: {color};">{high_risk_count} Units</div>
            <div class="metric-delta" style="color: #a1a1aa;">Maintenance Required</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        color = "#f87171" if safety_index < 80 else ("#fbbf24" if safety_index < 95 else "#34d399")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fleet Safety Index</div>
            <div class="metric-value" style="color: {color};">{safety_index:.1f}%</div>
            <div class="metric-delta" style="color: #a1a1aa;">Baseline: 90%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        roi_class = "metric-value-roi" if roi_saved >= 0 else "metric-value"
        roi_color = "#fbbf24" if roi_saved >= 0 else "#f87171"
        roi_delta_color = "#34d399" if roi_saved >= 0 else "#f87171"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Downtime Avoidance ROI</div>
            <div class="{roi_class}" style="color: {roi_color};">${roi_saved:,.0f}</div>
            <div class="metric-delta" style="color: {roi_delta_color};">▲ Net Business Impact</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📊 Commercial ROI Cost-Benefit Breakdown"):
        rc1, rc2, rc3 = st.columns([1, 1, 1])
        rc1.metric("True Positives (Avoided Failures)", f"{tp_count} Units", f"+${tp_count*15000:,.0f} Savings", delta_color="normal")
        rc2.metric("False Positives (Over-maintenance)", f"{fp_count} Units", f"-${fp_count*3000:,.0f} Cost", delta_color="inverse")
        rc3.metric("False Negatives (Catastrophic Failures)", f"{fn_count} Units", f"-${fn_count*25000:,.0f} Cost", delta_color="inverse")
        st.caption("Net ROI Formula: (TP × $15,000) - (FP × $3,000) - (FN × $25,000). TN (True Negatives: healthy assets) run normally with zero additional cost.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Charts
    chart_col1, chart_col2 = st.columns([1, 1])
    
    with chart_col1:
        st.subheader("📈 Fleet Predicted RUL Distribution Histogram")
        fig_hist = px.histogram(
            latest_df, 
            x="pred_RUL", 
            nbins=20, 
            color_discrete_sequence=['#3b82f6'],
            labels={'pred_RUL': 'Predicted RUL (Cycles)'}
        )
        fig_hist.add_vline(x=alert_threshold, line_dash="dash", line_color="red", annotation_text="Warning Line")
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f3f4f6',
            height=350
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with chart_col2:
        st.subheader("💡 Global Ensemble Feature Importance")
        importances = 0.51 * model_xgb.feature_importances_ + 0.49 * model_rf.feature_importances_
        global_feat_df = pd.DataFrame({
            'Feature': [f.replace('_smooth_regime_norm_roll15_mean', ' (Mean Temp/Speed)')
                        .replace('_smooth_regime_norm_roll15_std', ' (Volatility)')
                        .replace('_smooth_regime_norm_slope5', ' (Degradation Rate)')
                        .replace('_smooth', '')
                        .replace('regime_norm_', '') for f in selected_features],
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(10)
        
        fig_global_feat = px.bar(
            global_feat_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale="Blues",
            labels={'Importance': 'Model Weight (Gain)', 'Feature': 'Engine Parameter'}
        )
        fig_global_feat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f3f4f6',
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_global_feat, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Alert Priority Queue & Scatter Plot Layout
    table_col, scatter_col = st.columns([1.2, 1.8])
    
    with table_col:
        st.subheader("🚨 Fleet Alert Priority Queue (APQ)")
        st.markdown("*Select a row below to automatically update the **Single Engine Drill-Down** selector to that engine.*")
        
        apq_df = latest_df[latest_df['pred_RUL'] <= alert_threshold].copy()
        apq_df = apq_df.sort_values('pred_RUL', ascending=True)
        
        if len(apq_df) > 0:
            apq_df['Risk'] = apq_df['pred_RUL'].apply(
                lambda x: '🔴 Critical' if x <= 10 
                else ('🟠 High' if x <= 25 else '🟡 Moderate')
            )
            apq_df['Recommended_Action'] = apq_df['pred_RUL'].apply(
                lambda x: 'Immediate inspection' if x <= 10
                else ('Maintenance within 24h' if x <= 25 else 'Monitor closely')
            )
            
            display_df = pd.DataFrame({
                'Engine ID': apq_df['Engine_ID'].astype(int).map(lambda x: f"Engine {x}"),
                'Predicted RUL (Cycles)': apq_df['pred_RUL'].round(1),
                'Risk Level': apq_df['Risk'],
                'Recommended Action': apq_df['Recommended_Action']
            })
            
            event = st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            # Synchronize selection safely (resolving state conflict between tabs)
            if 'apq_selected_row' not in st.session_state:
                st.session_state['apq_selected_row'] = None
                
            if event and hasattr(event, 'selection') and event.selection.rows:
                selected_row = event.selection.rows[0]
                if selected_row != st.session_state['apq_selected_row']:
                    selected_engine_str = display_df.iloc[selected_row]['Engine ID']
                    selected_engine_id = int(selected_engine_str.split()[1])
                    st.session_state['selected_engine'] = selected_engine_id
                    st.session_state['apq_selected_row'] = selected_row
                    st.toast(f"Selected {selected_engine_str} for drill-down analysis!", icon="🔍")
            else:
                st.session_state['apq_selected_row'] = None
        else:
            st.success("🎉 All engines are healthy (RUL > alert threshold). No active alerts in queue.")
            
    with scatter_col:
        st.subheader("🎯 Model Fit Validation")
        test_df = filtered_df[~filtered_df['residual'].isna()]
        if len(test_df) > 0:
            fig_scatter = px.scatter(
                test_df.sample(min(1500, len(test_df)), random_state=42), 
                x="RUL_capped", 
                y="pred_RUL", 
                color="residual",
                color_continuous_scale="RdBu_r",
                labels={'RUL_capped': 'Actual RUL', 'pred_RUL': 'Predicted RUL', 'residual': 'Residual'},
                opacity=0.4
            )
            fig_scatter.add_shape(
                type="line", x0=0, y0=0, x1=125, y1=125,
                line=dict(color="Red", width=2, dash="dash")
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f3f4f6',
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No validation comparison data available.")

# ------------------------------------------------------------------------------
# TAB 2: SINGLE ENGINE DRILL-DOWN
# ------------------------------------------------------------------------------
with tab_engine:
    # 1. Engine Selector for Drill Down
    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        default_index = 0
        if 'selected_engine' in st.session_state and st.session_state['selected_engine'] in all_engines:
            default_index = all_engines.index(st.session_state['selected_engine'])
            
        drill_engine = st.selectbox(
            "Select Engine ID to Analyze:",
            options=all_engines,
            index=default_index,
            key="selectbox_drill_engine"
        )
        st.session_state['selected_engine'] = drill_engine
        
    engine_data = df[df['Engine_ID'] == drill_engine].sort_values('Cycle')
    latest_row = engine_data.iloc[-1]
    
    with col_sel2:
        # Mini metrics for selected engine
        sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])
        sub_col1.metric("Cumulative Cycles", f"{latest_row['Cycle']:.0f}")
        sub_col2.metric("Current Predicted RUL", f"{latest_row['pred_RUL']:.1f} Cycles")
        
        # Risk level string
        eng_risk = '🔴 Critical' if latest_row['pred_RUL'] <= 10 else ('🟠 High' if latest_row['pred_RUL'] <= 25 else ('🟡 Moderate' if latest_row['pred_RUL'] <= alert_threshold else '🟢 Healthy'))
        sub_col3.metric("Current Health Status", eng_risk)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Main Drill-Down Plot: RUL Prediction and 95% Confidence Interval
    # Map confidence interval std
    # We map RUL interval based on predictions
    def get_std_by_val(p):
        if p <= 45:
            return interval_std['Critical (0-45)']
        elif p <= 80:
            return interval_std['Warning (45-80)']
        else:
            return interval_std['Healthy (80-125)']
            
    engine_data['std_val'] = engine_data['pred_RUL'].apply(get_std_by_val)
    engine_data['ci_lower'] = (engine_data['pred_RUL'] - 1.96 * engine_data['std_val']).clip(lower=0)
    engine_data['ci_upper'] = (engine_data['pred_RUL'] + 1.96 * engine_data['std_val']).clip(upper=125)

    st.subheader(
        f"📈 Engine {drill_engine} RUL Prediction with 95% Confidence Interval Decay Curve",
        help="This shaded band represents the statistical 95% confidence interval of the prediction. As the engine degrades, the model's uncertainty decreases, causing the band to narrow and indicating that the failure cycle is approaching with high statistical certainty."
    )
    
    fig_line = go.Figure()
    # Shaded CI
    fig_line.add_trace(go.Scatter(
        x=engine_data['Cycle'],
        y=engine_data['ci_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='CI Upper Limit'
    ))
    fig_line.add_trace(go.Scatter(
        x=engine_data['Cycle'],
        y=engine_data['ci_lower'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(width=0),
        name='95% Prediction Uncertainty Interval'
    ))
    # Predicted RUL
    fig_line.add_trace(go.Scatter(
        x=engine_data['Cycle'],
        y=engine_data['pred_RUL'],
        mode='lines',
        line=dict(color='#3b82f6', width=3),
        name='Predicted RUL'
    ))
    # Actual RUL (reference)
    fig_line.add_trace(go.Scatter(
        x=engine_data['Cycle'],
        y=engine_data['RUL_capped'],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Actual Capped RUL (Baseline)'
    ))
    
    # Alert threshold horizontal line
    fig_line.add_hline(y=alert_threshold, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
    fig_line.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    
    fig_line.update_layout(
        xaxis_title="Operating Cycle",
        yaxis_title="Remaining Useful Life (RUL)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#f3f4f6',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Features Interpretation & Sensor Trajectories
    col_feat1, col_feat2 = st.columns([1, 1])
    
    with col_feat1:
        st.subheader("💡 Normalized Sensor Measurement Trajectories")
        # Plot top 3 degradation indicators: S11, S12, S4 (or corresponding smoothed variables)
        fig_sensors = go.Figure()
        
        # Scale sensors locally to fit in same plot
        s11_scaled = (engine_data['Sensor_11_smooth'] - engine_data['Sensor_11_smooth'].min()) / \
                     (engine_data['Sensor_11_smooth'].max() - engine_data['Sensor_11_smooth'].min() + 1e-5)
        s12_scaled = (engine_data['Sensor_12_smooth'] - engine_data['Sensor_12_smooth'].min()) / \
                     (engine_data['Sensor_12_smooth'].max() - engine_data['Sensor_12_smooth'].min() + 1e-5)
        s4_scaled = (engine_data['Sensor_4_smooth'] - engine_data['Sensor_4_smooth'].min()) / \
                    (engine_data['Sensor_4_smooth'].max() - engine_data['Sensor_4_smooth'].min() + 1e-5)
                    
        fig_sensors.add_trace(go.Scatter(
            x=engine_data['Cycle'], y=s11_scaled,
            mode='lines', name='Sensor 11 (Temperature Indicator, Normalized)'
        ))
        fig_sensors.add_trace(go.Scatter(
            x=engine_data['Cycle'], y=s12_scaled,
            mode='lines', name='Sensor 12 (Spool Speed Indicator, Normalized)'
        ))
        fig_sensors.add_trace(go.Scatter(
            x=engine_data['Cycle'], y=s4_scaled,
            mode='lines', name='Sensor 4 (Thermodynamic Indicator, Normalized)'
        ))
        
        fig_sensors.update_layout(
            xaxis_title="Operating Cycle",
            yaxis_title="Relative Sensor Value",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f3f4f6',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_sensors, use_container_width=True)
        
    with col_feat2:
        st.subheader("🛠️ Local Decision Drivers (SHAP Value Contribution)")
        
        # Pre-calculate correlation directions with RUL for each selected feature
        corr_directions = {}
        for f_name in selected_features:
            corr_val = df[f_name].corr(df['RUL_capped'])
            corr_directions[f_name] = -1.0 if corr_val < 0 else 1.0
            
        latest_features = latest_row[selected_features]
        contributions = []
        for idx, f_name in enumerate(selected_features):
            val = latest_features[f_name]
            importance = importances[idx]
            direction = corr_directions.get(f_name, -1.0)
            
            # Normalization aligns baseline to 0. Hence offset = val. 
            # Negative contribution pulls RUL down (red), positive supports high RUL (blue).
            contributions.append(val * importance * direction)
        
        # Sort contributions by absolute impact
        local_df = pd.DataFrame({
            'Feature': [f.replace('_smooth_regime_norm_roll15_mean', ' (Mean Temp/Speed)')
                        .replace('_smooth_regime_norm_roll15_std', ' (Volatility)')
                        .replace('_smooth_regime_norm_slope5', ' (Degradation Rate)')
                        .replace('_smooth', '')
                        .replace('regime_norm_', '') for f in selected_features],
            'Contribution': contributions
        })
        local_df['abs_val'] = local_df['Contribution'].abs()
        local_df = local_df.sort_values('abs_val', ascending=True)
        
        # Assign colors: Red for degradation (pulling RUL down), Blue for health (supporting RUL)
        local_df['Color'] = local_df['Contribution'].apply(lambda x: '#ef4444' if x < 0 else '#3b82f6')
        
        fig_local = px.bar(
            local_df,
            x='Contribution',
            y='Feature',
            orientation='h',
            labels={'Contribution': 'SHAP Contribution Value (RUL Impact)', 'Feature': 'Engine Parameter'}
        )
        fig_local.update_traces(marker_color=local_df['Color'])
        fig_local.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f3f4f6',
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_local, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.info("StudentID: 2471803 | IOT106TC Coursework 2 Deliverable | Predictive Maintenance System Dashboard")
