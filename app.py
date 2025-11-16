"""
Streamlit Web Application for AML Transaction Anomaly Detector
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from aml_detector import AMLDetector
from explainability_engine import ExplainabilityEngine
import os


# Page configuration
st.set_page_config(
    page_title="AML Transaction Anomaly Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .critical-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .high-alert {
        background-color: #ffa500;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .medium-alert {
        background-color: #ffed4e;
        color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = AMLDetector()
    st.session_state.is_trained = False
    st.session_state.results = None
    st.session_state.training_data = None


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🔍 AML Transaction Anomaly Detector</div>', 
                unsafe_allow_html=True)
    st.markdown("**AI-Powered Anti-Money Laundering Detection System**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔬 Analysis", "📈 Visualizations", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.header("📁 Data Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Transaction CSV",
            type=['csv'],
            help="Upload a CSV file with columns: date, amount, merchant, country, channel"
        )
        
        # Sample data option
        if st.button("📥 Use Sample Data"):
            if os.path.exists('transactions_training.csv'):
                st.session_state.training_data = pd.read_csv('transactions_training.csv')
                st.success("✓ Sample data loaded!")
            else:
                st.warning("Sample data not found. Please generate it first.")
        
        # Load uploaded file
        if uploaded_file is not None:
            st.session_state.training_data = pd.read_csv(uploaded_file)
            st.success(f"✓ Loaded {len(st.session_state.training_data)} transactions")
        
        # Training
        if st.session_state.training_data is not None and not st.session_state.is_trained:
            if st.button("🎯 Train System"):
                with st.spinner("Training AML detection system..."):
                    st.session_state.detector.transactions_df = st.session_state.training_data
                    st.session_state.detector.train()
                    st.session_state.is_trained = True
                    st.success("✓ System trained successfully!")
                    st.rerun()
        
        # Analysis
        if st.session_state.is_trained:
            st.markdown("---")
            st.header("🔍 Run Analysis")
            
            analysis_file = st.file_uploader(
                "Upload Transactions to Analyze",
                type=['csv'],
                key='analysis_file'
            )
            
            if analysis_file is not None:
                if st.button("🚀 Analyze Transactions"):
                    with st.spinner("Analyzing transactions..."):
                        test_data = pd.read_csv(analysis_file)
                        results = st.session_state.detector.analyze_batch(test_data)
                        st.session_state.results = results
                        st.success(f"✓ Analyzed {len(results)} transactions")
                        st.rerun()
            
            # Use test data
            if os.path.exists('transactions_test.csv'):
                if st.button("📊 Analyze Test Data"):
                    with st.spinner("Analyzing test transactions..."):
                        test_data = pd.read_csv('transactions_test.csv')
                        results = st.session_state.detector.analyze_batch(test_data)
                        st.session_state.results = results
                        st.success(f"✓ Analyzed {len(results)} transactions")
                        st.rerun()
    
    # Main content based on page selection
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "🔬 Analysis":
        show_analysis()
    elif page == "📈 Visualizations":
        show_visualizations()
    elif page == "ℹ️ About":
        show_about()


def show_dashboard():
    """Dashboard page with overview metrics"""
    
    st.header("📊 Dashboard Overview")
    
    if st.session_state.results is None:
        st.info("👈 Please upload data and run analysis from the sidebar to see results.")
        
        # Show system status
        col1, col2 = st.columns(2)
        with col1:
            st.metric("System Status", 
                     "✓ Trained" if st.session_state.is_trained else "⚠️ Not Trained")
        with col2:
            if st.session_state.training_data is not None:
                st.metric("Training Data", f"{len(st.session_state.training_data)} transactions")
        return
    
    results = st.session_state.results
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(results)
    flagged = results[results['is_flagged'] == True]
    num_flagged = len(flagged)
    
    with col1:
        st.metric("Total Transactions", f"{total:,}")
    
    with col2:
        st.metric("Flagged", f"{num_flagged:,}", 
                 delta=f"{num_flagged/total*100:.1f}%" if total > 0 else "0%")
    
    with col3:
        critical = len(results[results['severity'] == 'critical'])
        st.metric("Critical Alerts", f"{critical:,}", 
                 delta="⚠️" if critical > 0 else "✓")
    
    with col4:
        avg_risk = results['risk_score'].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.2f}")
    
    st.markdown("---")
    
    # Severity distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Severity Distribution")
        severity_counts = results['severity'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=severity_counts.index,
            values=severity_counts.values,
            hole=0.4,
            marker=dict(colors=['#ff4b4b', '#ffa500', '#ffed4e', '#90EE90'])
        )])
        fig.update_layout(height=350, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Score Distribution")
        fig = go.Figure(data=[go.Histogram(
            x=results['risk_score'],
            nbinsx=20,
            marker_color='#1f77b4'
        )])
        fig.update_layout(
            xaxis_title="Risk Score",
            yaxis_title="Count",
            height=350,
            margin=dict(t=30, b=50, l=50, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top alerts
    st.markdown("---")
    st.subheader("🚨 Top 10 Highest Risk Transactions")
    
    top_alerts = flagged.nlargest(10, 'risk_score') if len(flagged) > 0 else pd.DataFrame()
    
    if len(top_alerts) > 0:
        display_cols = ['transaction_id', 'customer_id', 'amount', 'country', 
                       'risk_score', 'severity', 'num_rule_violations']
        
        # Format the dataframe
        display_df = top_alerts[display_cols].copy()
        display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
        display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.success("No high-risk transactions detected!")


def show_analysis():
    """Detailed analysis page"""
    
    st.header("🔬 Detailed Transaction Analysis")
    
    if st.session_state.results is None:
        st.info("👈 Please run analysis from the sidebar first.")
        return
    
    results = st.session_state.results
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=['critical', 'high', 'medium', 'low'],
            default=['critical', 'high', 'medium']
        )
    
    with col2:
        min_risk = st.slider("Minimum Risk Score", 0.0, 1.0, 0.5, 0.05)
    
    with col3:
        customer_filter = st.multiselect(
            "Filter by Customer",
            options=results['customer_id'].unique().tolist(),
            default=results['customer_id'].unique().tolist()
        )
    
    # Apply filters
    filtered = results[
        (results['severity'].isin(severity_filter)) &
        (results['risk_score'] >= min_risk) &
        (results['customer_id'].isin(customer_filter))
    ].sort_values('risk_score', ascending=False)
    
    st.markdown(f"**Showing {len(filtered)} transactions**")
    
    # Transaction details
    if len(filtered) > 0:
        for idx, row in filtered.head(20).iterrows():
            with st.expander(
                f"🔍 {row['transaction_id']} - {row['severity'].upper()} "
                f"(Risk: {row['risk_score']:.2f})"
            ):
                show_transaction_detail(row)
    else:
        st.info("No transactions match the selected filters.")


def show_transaction_detail(transaction):
    """Show detailed information for a single transaction"""
    
    # Alert box based on severity
    severity = transaction['severity']
    if severity == 'critical':
        st.markdown(f'<div class="critical-alert">⚠️ CRITICAL RISK - Immediate Action Required</div>', 
                   unsafe_allow_html=True)
    elif severity == 'high':
        st.markdown(f'<div class="high-alert">⚠️ HIGH RISK - Urgent Review Needed</div>', 
                   unsafe_allow_html=True)
    elif severity == 'medium':
        st.markdown(f'<div class="medium-alert">⚠️ MEDIUM RISK - Review Recommended</div>', 
                   unsafe_allow_html=True)
    
    # Transaction details
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Amount", f"${transaction['amount']:,.2f}")
    with col2:
        st.metric("Country", transaction['country'])
    with col3:
        st.metric("Channel", transaction['channel'])
    with col4:
        st.metric("Risk Score", f"{transaction['risk_score']:.2f}")
    
    # Reasons
    st.markdown("**🎯 Why This Transaction Was Flagged:**")
    
    if 'primary_reasons' in transaction and isinstance(transaction['primary_reasons'], list):
        for i, reason in enumerate(transaction['primary_reasons'], 1):
            if isinstance(reason, dict):
                st.markdown(f"{i}. **{reason.get('explanation', 'N/A')}**")
                st.markdown(f"   - Category: `{reason.get('category', 'N/A')}`")
                st.markdown(f"   - Score: `{reason.get('score', 0):.2f}`")
    
    # ML Detection
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🤖 ML Detection:**")
        st.markdown(f"- Models Flagged: {transaction.get('num_ml_models_flagged', 0)} / 3")
        st.markdown(f"- ML Confidence: {transaction.get('ml_confidence', 0):.2f}")
    
    with col2:
        st.markdown("**📋 Rule Violations:**")
        st.markdown(f"- Total Violations: {transaction.get('num_rule_violations', 0)}")
    
    # Recommendation
    if 'recommendation' in transaction:
        st.markdown("**💡 Recommendation:**")
        st.info(transaction['recommendation'])


def show_visualizations():
    """Visualizations page"""
    
    st.header("📈 Data Visualizations")
    
    if st.session_state.results is None:
        st.info("👈 Please run analysis from the sidebar first.")
        return
    
    results = st.session_state.results
    
    # Time series
    st.subheader("📅 Transactions Over Time")
    results_copy = results.copy()
    results_copy['date'] = pd.to_datetime(results_copy['date'])
    daily_stats = results_copy.groupby(results_copy['date'].dt.date).agg({
        'transaction_id': 'count',
        'is_flagged': 'sum',
        'risk_score': 'mean'
    }).reset_index()
    daily_stats.columns = ['date', 'total', 'flagged', 'avg_risk']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=daily_stats['date'], y=daily_stats['total'], name="Total Transactions",
               marker_color='lightblue'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(x=daily_stats['date'], y=daily_stats['flagged'], name="Flagged",
               marker_color='red'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=daily_stats['date'], y=daily_stats['avg_risk'], name="Avg Risk Score",
                  mode='lines+markers', marker_color='orange'),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
    fig.update_yaxes(title_text="Risk Score", secondary_y=True)
    fig.update_layout(height=400, margin=dict(t=30, b=50, l=50, r=50))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Geographic distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Transactions by Country")
        country_stats = results.groupby('country').agg({
            'transaction_id': 'count',
            'risk_score': 'mean'
        }).reset_index()
        country_stats.columns = ['country', 'count', 'avg_risk']
        country_stats = country_stats.nlargest(10, 'count')
        
        fig = go.Figure(data=[go.Bar(
            x=country_stats['country'],
            y=country_stats['count'],
            marker_color=country_stats['avg_risk'],
            marker_colorscale='Reds',
            text=country_stats['count'],
            textposition='auto'
        )])
        fig.update_layout(
            xaxis_title="Country",
            yaxis_title="Transaction Count",
            height=350,
            margin=dict(t=30, b=50, l=50, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💳 Transactions by Channel")
        channel_stats = results.groupby('channel').agg({
            'transaction_id': 'count',
            'risk_score': 'mean'
        }).reset_index()
        channel_stats.columns = ['channel', 'count', 'avg_risk']
        
        fig = go.Figure(data=[go.Bar(
            x=channel_stats['channel'],
            y=channel_stats['count'],
            marker_color=channel_stats['avg_risk'],
            marker_colorscale='Oranges',
            text=channel_stats['count'],
            textposition='auto'
        )])
        fig.update_layout(
            xaxis_title="Channel",
            yaxis_title="Transaction Count",
            height=350,
            margin=dict(t=30, b=50, l=50, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Amount analysis
    st.subheader("💰 Transaction Amount Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount vs Risk Score
        fig = px.scatter(
            results,
            x='amount',
            y='risk_score',
            color='severity',
            color_discrete_map={
                'low': '#90EE90',
                'medium': '#ffed4e',
                'high': '#ffa500',
                'critical': '#ff4b4b'
            },
            hover_data=['transaction_id', 'customer_id', 'country'],
            title="Amount vs Risk Score"
        )
        fig.update_layout(height=350, margin=dict(t=50, b=50, l=50, r=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot by severity
        fig = px.box(
            results,
            x='severity',
            y='amount',
            color='severity',
            color_discrete_map={
                'low': '#90EE90',
                'medium': '#ffed4e',
                'high': '#ffa500',
                'critical': '#ff4b4b'
            },
            title="Amount Distribution by Severity"
        )
        fig.update_layout(height=350, margin=dict(t=50, b=50, l=50, r=0))
        st.plotly_chart(fig, use_container_width=True)


def show_about():
    """About page"""
    
    st.header("ℹ️ About AML Transaction Anomaly Detector")
    
    st.markdown("""
    ### 🎯 Overview
    
    This is a comprehensive **Anti-Money Laundering (AML) Transaction Anomaly Detection System** 
    that combines multiple detection approaches:
    
    - **Machine Learning Models**: Isolation Forest, One-Class SVM, and Local Outlier Factor
    - **Rule-Based Detection**: High-risk countries, structuring, rapid movement, and more
    - **Customer Baseline Analysis**: Compares transactions to customer's normal behavior
    - **Explainability**: Provides clear reasons why transactions are flagged
    
    ---
    
    ### 🔍 Detection Methods
    
    #### 1. Machine Learning Detection
    - **Isolation Forest**: Detects anomalies by isolating outliers
    - **One-Class SVM**: Learns the boundary of normal behavior
    - **Local Outlier Factor**: Identifies local density deviations
    
    #### 2. Rule-Based Detection
    - **High-Risk Countries**: Flags transactions from sanctioned/high-risk jurisdictions
    - **Structuring**: Detects multiple transactions just below reporting thresholds
    - **Rapid Movement**: Identifies unusual velocity of transactions
    - **Large Transactions**: Flags amounts significantly above customer baseline
    - **Round Amounts**: Detects suspiciously round large amounts
    - **Unusual Channels**: Identifies rarely-used transaction methods
    
    #### 3. Baseline Analysis
    - Computes customer-specific patterns (amounts, locations, merchants, times)
    - Measures deviations from normal behavior
    - Adapts to each customer's unique profile
    
    ---
    
    ### 📊 Risk Scoring
    
    Each transaction receives a **risk score (0-1)** based on:
    - 30% Baseline deviations
    - 40% Rule violations
    - 30% ML anomaly scores
    
    **Severity Levels:**
    - 🟢 **Low** (0.0-0.3): Minor deviations, log for records
    - 🟡 **Medium** (0.3-0.5): Review recommended within 24 hours
    - 🟠 **High** (0.5-0.7): Urgent review required
    - 🔴 **Critical** (0.7-1.0): Immediate action, potential money laundering
    
    ---
    
    ### 🛠️ Technology Stack
    
    - **Backend**: Python, scikit-learn, pandas, numpy
    - **Frontend**: Streamlit
    - **Visualization**: Plotly
    - **ML Models**: Isolation Forest, One-Class SVM, LOF
    
    ---
    
    ### 📝 How to Use
    
    1. **Upload Training Data**: CSV with historical transactions
    2. **Train System**: Click "Train System" to build customer baselines and ML models
    3. **Analyze Transactions**: Upload new transactions to analyze
    4. **Review Results**: Examine flagged transactions with detailed explanations
    5. **Export**: Download results for compliance reporting
    
    ---
    
    ### 📄 Required CSV Format
    
    Your CSV file should contain these columns:
    - `date`: Transaction date/time (YYYY-MM-DD HH:MM:SS)
    - `amount`: Transaction amount (numeric)
    - `merchant`: Merchant name (text)
    - `country`: Country code or name (text)
    - `channel`: Transaction channel (online, pos, atm, mobile, etc.)
    - `customer_id`: Customer identifier (optional, will be auto-generated)
    
    ---
    
    ### ⚖️ Compliance Note
    
    This system is designed for educational and demonstration purposes. 
    For production use in financial institutions, additional regulatory 
    requirements, audit trails, and compliance features would be needed.
    
    ---
    
    ### 👨‍💻 Developer
    
    Built with ❤️ using Python, scikit-learn, and Streamlit
    """)


if __name__ == '__main__':
    main()

