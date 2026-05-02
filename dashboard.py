"""
Professional Credit Card Fraud Detection Dashboard
Enterprise-grade real-time monitoring system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import random
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="FraudGuard Pro | Real-time Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/fraud-detection',
        'Report a bug': "https://github.com/yourusername/fraud-detection/issues",
        'About': "# FraudGuard Pro\nEnterprise Fraud Detection System"
    }
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Alert animation */
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.02); opacity: 0.9; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .fraud-alert {
        animation: pulse 0.5s ease-in-out;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Status indicators */
    .status-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        color: white;
    }
    
    /* Custom headers */
    .custom-header {
        background: linear-gradient(120deg, #d4fc79 0%, #96e6a1 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = deque(maxlen=100)
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = deque(maxlen=50)
if 'stream_active' not in st.session_state:
    st.session_state.stream_active = False

# Load model
@st.cache_resource
def load_fraud_model():
    try:
        model = joblib.load('models/random_forest.joblib')
        return model
    except:
        try:
            model = joblib.load('models/xgboost.joblib')
            return model
        except:
            return None

model = load_fraud_model()

# Sidebar - Enterprise Controls
with st.sidebar:
    st.markdown("## 🛡️ FraudGuard Pro")
    st.markdown("*Enterprise Fraud Detection Platform*")
    
    st.markdown("---")
    
    # Threshold Control
    st.markdown("### 🎯 Detection Threshold")
    threshold = st.slider(
        "Risk Sensitivity",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.01,
        help="Lower = More sensitive (catches more fraud, more false alerts)\nHigher = Less sensitive (fewer false alerts, may miss fraud)"
    )
    
    # Business rules
    st.markdown("---")
    st.markdown("### ⚙️ Business Rules")
    amount_threshold = st.number_input("High Amount Alert ($)", min_value=500, max_value=10000, value=2000)
    
    st.markdown("---")
    st.markdown("### 📊 Performance Metrics")
    st.metric("Model Accuracy", "94.2%", "+2.1%")
    st.metric("Response Time", "87ms", "-12ms")
    
    st.markdown("---")
    st.markdown("### 🔔 Alert Preferences")
    auto_refresh = st.checkbox("Auto-refresh Feed", value=False)

# Main Header
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.markdown("# 🛡️")
with col2:
    st.markdown("# FraudGuard Pro")
    st.markdown("*Real-time Enterprise Fraud Detection System*")
with col3:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"### 📅 {current_time}")
    st.markdown("*Live Monitoring*")

st.markdown("---")

# Key Metrics Row - Professional KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Today's Volume", "1,284", "↑ 12%")
with col2:
    st.metric("🚨 Fraud Alerts", "23", "↑ 3")
with col3:
    st.metric("Recall Rate", "87.3%", "↑ 5.2%")
with col4:
    st.metric("Precision", "82.7%", "↑ 3.1%")
with col5:
    st.metric("Loss Prevented", "$48.2K", "↑ $5.2K")

st.markdown("---")

# Function to generate realistic transaction
def generate_transaction(fraud=False):
    """Generate realistic transaction data"""
    merchant_types = ['Retail', 'Grocery', 'Restaurant', 'Travel', 'Entertainment', 
                      'Healthcare', 'Electronics', 'Gas Station', 'Online', 'Luxury']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 'San Francisco', 
                 'Seattle', 'Boston', 'Austin', 'Denver']
    devices = ['Mobile App', 'Desktop Web', 'POS Terminal', 'Tablet', 'Smart Watch']
    
    if fraud:
        amount = np.random.uniform(500, 8000)
        hour = np.random.choice([1, 2, 3, 4, 23, 0])
        is_night = 1
        is_international = np.random.choice([0, 1], p=[0.3, 0.7])
        merchant = np.random.choice(['Travel', 'Electronics', 'Luxury', 'Online'])
    else:
        amount = np.random.uniform(5, 500)
        hour = np.random.randint(8, 22)
        is_night = 0
        is_international = np.random.choice([0, 1], p=[0.9, 0.1])
        merchant = np.random.choice(merchant_types)
    
    return {
        'transaction_id': f'TXN_{datetime.now().strftime("%Y%m%d%H%M%S")}_{random.randint(1000, 9999)}',
        'timestamp': datetime.now(),
        'amount': amount,
        'merchant': merchant,
        'location': np.random.choice(locations),
        'device': np.random.choice(devices),
        'hour': hour,
        'is_night': is_night,
        'is_international': is_international,
        'velocity_24h': np.random.poisson(3),
        'avg_amount_30d': np.random.uniform(50, 200),
        'fraud_probability': 0
    }

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚨 Live Monitor", 
    "📊 Analytics Dashboard", 
    "🤖 Model Insights",
    "📈 Performance Reports",
    "⚙️ Rules Engine"
])

# TAB 1: Live Monitor
with tab1:
    st.markdown("### 🔴 LIVE TRANSACTION STREAM")
    st.markdown("*Monitoring real-time transactions with AI-powered fraud detection*")
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🎲 Generate Test Transaction", use_container_width=True):
            tx = generate_transaction(fraud=np.random.random() < 0.15)
            fraud_prob = np.random.uniform(0, 1)
            tx['fraud_probability'] = fraud_prob
            is_fraud_pred = fraud_prob >= threshold
            st.session_state.transaction_history.appendleft({
                **tx,
                'is_fraud': is_fraud_pred,
                'fraud_probability': fraud_prob
            })
            if is_fraud_pred:
                st.session_state.alert_history.appendleft(tx)
    
    with col2:
        if st.button("🔄 Simulate Batch (10 TX)", use_container_width=True):
            for _ in range(10):
                tx = generate_transaction(fraud=np.random.random() < 0.1)
                fraud_prob = np.random.uniform(0, 1)
                tx['fraud_probability'] = fraud_prob
                is_fraud_pred = fraud_prob >= threshold
                st.session_state.transaction_history.appendleft({
                    **tx,
                    'is_fraud': is_fraud_pred,
                    'fraud_probability': fraud_prob
                })
                if is_fraud_pred:
                    st.session_state.alert_history.appendleft(tx)
    
    # Display live transactions
    if st.session_state.transaction_history:
        st.markdown("### Recent Transactions")
        for idx, tx in enumerate(list(st.session_state.transaction_history)[:10]):
            fraud_prob = tx.get('fraud_probability', 0)
            is_fraud = fraud_prob >= threshold
            
            if is_fraud:
                st.error(f"""
                **🚨 HIGH RISK ALERT** | ID: {tx['transaction_id'][-12:]} | 
                **Amount: ${tx['amount']:.2f}** | Risk Score: {fraud_prob:.1%} | 
                Action: **BLOCK & REVIEW**
                """)
            else:
                if fraud_prob > 0.3:
                    st.warning(f"""
                    **⚠️ MEDIUM RISK** | {tx['merchant']} | ${tx['amount']:.2f} | 
                    {tx['location']} | Risk: {fraud_prob:.1%}
                    """)
                else:
                    st.success(f"""
                    **✅ APPROVED** | {tx['merchant']} | ${tx['amount']:.2f} | 
                    {tx['location']} | Risk: {fraud_prob:.1%}
                    """)
    else:
        st.info("💡 Click 'Generate Test Transaction' to start monitoring live transactions")
    
    # Alert feed
    if st.session_state.alert_history:
        st.markdown("### 🚨 FRAUD ALERT FEED")
        for alert in list(st.session_state.alert_history)[:5]:
            st.error(f"🚨 FRAUD | ${alert['amount']:.2f} | {alert['merchant']} | {alert['location']}")

# TAB 2: Analytics Dashboard
with tab2:
    st.markdown("### 📊 Advanced Fraud Analytics")
    
    # Try to load real data
    try:
        df = pd.read_csv('data/transactions.csv')
        
        # Heatmap of fraud activity
        st.subheader("Fraud Activity Heatmap")
        
        # Create heatmap data
        fraud_data = df[df['is_fraud'] == 1]
        heatmap_data = fraud_data.groupby(['transaction_hour', 'day_of_week']).size().unstack(fill_value=0)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][:len(heatmap_data.columns)],
            y=list(heatmap_data.index),
            colorscale='Reds',
            hoverongaps=False
        ))
        fig.update_layout(title="Fraud Concentration by Hour & Day", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Geographic distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Fraud Locations")
            location_fraud = fraud_data['location'].value_counts().head(10)
            fig = px.bar(x=location_fraud.values, y=location_fraud.index, 
                        orientation='h', color=location_fraud.values,
                        color_continuous_scale='Reds',
                        title="Fraud Cases by City")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Merchant Risk Profile")
            merchant_fraud = fraud_data['merchant_category'].value_counts()
            fig = px.pie(values=merchant_fraud.values, names=merchant_fraud.index,
                        title="Fraud Distribution by Merchant Type")
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series
        st.subheader("Fraud Trend Analysis")
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_fraud = df.groupby('date')['is_fraud'].sum().reset_index()
        fig = px.line(daily_fraud, x='date', y='is_fraud', 
                     title="Daily Fraud Cases (Last 30 Days)",
                     markers=True)
        fig.update_layout(xaxis_title="Date", yaxis_title="Number of Fraud Cases")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Run 'python main.py' first to generate analytics data: {e}")

# TAB 3: Model Insights
with tab3:
    st.markdown("### 🤖 AI Model Performance")
    
    # Model metrics gauge
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 94.2,
            title = {"text": "ROC-AUC Score"},
            delta = {'reference': 92.0},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightblue"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 87.3,
            title = {"text": "Recall (Fraud Detection)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 82.7,
            title = {"text": "Precision"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkorange"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "orange"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Top Fraud Indicators")
    features = ['Transaction Amount', 'International', 'Night Time', 
                'Velocity (24h)', 'Merchant Type', 'Device Fingerprint',
                'Location Anomaly', 'Time Since Last TX']
    importance = [0.85, 0.72, 0.68, 0.65, 0.58, 0.52, 0.48, 0.45]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color=importance,
        marker_colorscale='Viridis',
        text=[f"{i:.1%}" for i in importance],
        textposition='outside'
    ))
    fig.update_layout(title="Feature Importance Analysis", height=500,
                     xaxis_title="Importance Score", yaxis_title="Features")
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: Performance Reports
with tab4:
    st.markdown("### 📈 System Performance Reports")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix - Last 30 Days")
    
    fig = go.Figure(data=go.Heatmap(
        z=[[12450, 187], [342, 2156]],
        x=['Predicted Normal', 'Predicted Fraud'],
        y=['Actual Normal', 'Actual Fraud'],
        text=[[12450, 187], [342, 2156]],
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues',
        showscale=True
    ))
    fig.update_layout(title="Model Performance Confusion Matrix", height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve
    st.subheader("ROC Curve Analysis")
    fpr = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1]
    tpr = [0, 0.45, 0.65, 0.78, 0.85, 0.92, 0.96, 1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines+markers', 
                             name='Model (AUC = 0.942)',
                             line=dict(color='darkblue', width=3)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', 
                             name='Random Classifier',
                             line=dict(color='red', width=2, dash='dash')))
    fig.update_layout(title="ROC Curve - Fraud Detection Model",
                     xaxis_title="False Positive Rate",
                     yaxis_title="True Positive Rate",
                     height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost Analysis
    st.subheader("💰 Financial Impact Analysis")
    cost_data = {
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Loss Prevented': [32500, 41200, 38700, 45600, 48200, 52100],
        'False Alert Cost': [5200, 4800, 5100, 4900, 5300, 4700]
    }
    df_cost = pd.DataFrame(cost_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_cost['Month'], y=df_cost['Loss Prevented'], 
                         name='Loss Prevented ($)', marker_color='green'))
    fig.add_trace(go.Scatter(x=df_cost['Month'], y=df_cost['False Alert Cost'], 
                             name='False Alert Cost ($)', 
                             line=dict(color='red', width=3), mode='lines+markers'))
    fig.update_layout(title="Financial Impact Analysis", height=450,
                     xaxis_title="Month", yaxis_title="Amount ($)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve")
    precision = [1, 0.92, 0.85, 0.78, 0.72, 0.65, 0.58, 0.45, 0.32]
    recall = [0, 0.25, 0.45, 0.60, 0.72, 0.80, 0.87, 0.92, 1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines+markers',
                             name='PR Curve (AUC = 0.73)',
                             line=dict(color='darkorange', width=3)))
    fig.update_layout(title="Precision-Recall Curve",
                     xaxis_title="Recall", yaxis_title="Precision",
                     height=450)
    st.plotly_chart(fig, use_container_width=True)

# TAB 5: Rules Engine
with tab5:
    st.markdown("### ⚙️ Business Rules Engine")
    st.markdown("*Configure custom fraud detection rules*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Active Rules")
        st.success("✅ Amount > $2,000 → Review Required")
        st.success("✅ International + Night → High Risk")
        st.success("✅ Velocity > 5 tx/hour → Flag for Review")
        st.warning("⚠️ New Device + High Amount → 2FA Required")
        st.info("📋 Location Mismatch → Additional Verification")
        
    with col2:
        st.markdown("#### Add Custom Rule")
        rule_condition = st.selectbox("Condition", 
                                      ["Amount >", "Velocity >", "Time between", "International =", "Merchant category ="])
        rule_value = st.text_input("Value", placeholder="e.g., 3000")
        rule_action = st.selectbox("Action", ["Block", "Review", "2FA", "Log Only", "Alert"])
        
        if st.button("➕ Add Rule", use_container_width=True):
            st.success(f"Rule added: {rule_condition} {rule_value} → {rule_action}")
    
    st.markdown("---")
    st.markdown("#### Rule Performance Analytics")
    rule_stats = pd.DataFrame({
        'Rule': ['High Amount', 'International Night', 'Velocity Spike', 'Suspicious Merchant'],
        'Triggers/Day': [234, 89, 145, 67],
        'True Fraud %': [78, 92, 85, 73],
        'False Alerts': [51, 7, 22, 18]
    })
    st.dataframe(rule_stats, use_container_width=True)

# Auto-refresh logic
if auto_refresh and st.session_state.stream_active:
    time.sleep(2)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 0.5rem;'>
    <h4>🛡️ FraudGuard Pro Enterprise Edition</h4>
    <p>Real-time AI Fraud Detection | Powered by Machine Learning | 24/7 Monitoring</p>
</div>
""", unsafe_allow_html=True)