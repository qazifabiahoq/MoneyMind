import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

try:
    from groq import Groq
    HAS_GROQ = True
except:
    HAS_GROQ = False

# Page configuration
st.set_page_config(
    page_title="MoneyMind Pro - ML-Powered Finance",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-blue: #1E3A8A;
        --accent-green: #10B981;
        --accent-purple: #8B5CF6;
        --accent-orange: #F59E0B;
        --light-bg: #F9FAFB;
        --card-bg: #FFFFFF;
        --text-dark: #1F2937;
        --text-light: #6B7280;
        --border-color: #E5E7EB;
        --success-green: #059669;
        --warning-red: #DC2626;
    }
    
    .stApp {
        background: var(--light-bg);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    body, p, div, span, label, input, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-dark);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: var(--text-dark);
    }
    
    .main-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, #1E40AF 100%);
        padding: 2.5rem 2rem;
        border-radius: 0;
        margin: -6rem -5rem 2rem -5rem;
        text-align: center;
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .app-subtitle {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    
    .app-tagline {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .metric-container {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
        transition: all 0.2s ease;
    }
    
    .metric-container:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-light);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-blue);
        margin: 0.5rem 0;
    }
    
    .metric-value.positive {
        color: var(--success-green);
    }
    
    .metric-value.negative {
        color: var(--warning-red);
    }
    
    .metric-subtext {
        font-size: 0.8rem;
        color: var(--text-light);
        margin-top: 0.25rem;
    }
    
    .ml-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--accent-purple) 0%, #A78BFA 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-left: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue) 0%, #1E40AF 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        width: 100%;
        height: 50px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1E40AF 0%, #2563EB 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 4px solid var(--primary-blue);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .insight-box h4 {
        color: var(--primary-blue);
        margin-bottom: 0.75rem;
    }
    
    .model-metrics {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .model-metrics h4 {
        color: var(--accent-purple);
        margin-bottom: 1rem;
    }
    
    [data-testid="stSidebar"] {
        background: var(--card-bg);
        border-right: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-blue);
        color: #FFFFFF;
        border-color: var(--primary-blue);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ml_models_trained' not in st.session_state:
    st.session_state.ml_models_trained = False

# ML Functions
class MoneyMindML:
    """Machine Learning engine for financial analysis"""
    
    @staticmethod
    def engineer_features(df):
        """Feature engineering for ML models"""
        df = df.copy()
        
        # Convert date column
        date_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time']):
                date_col = col
                break
        
        if date_col:
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Find amount column
        amount_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['amount', 'value', 'total', 'price']):
                amount_col = col
                break
        
        if amount_col:
            df['amount'] = pd.to_numeric(df[amount_col], errors='coerce')
            
            # Rolling statistics
            df['rolling_mean_7d'] = df['amount'].rolling(window=7, min_periods=1).mean()
            df['rolling_std_7d'] = df['amount'].rolling(window=7, min_periods=1).std()
            df['rolling_max_7d'] = df['amount'].rolling(window=7, min_periods=1).max()
            
            # Spending velocity
            df['spending_velocity'] = df['amount'].diff().fillna(0)
            
            # Cumulative spending
            df['cumulative_spending'] = df['amount'].cumsum()
            
            # Z-score for anomaly detection
            df['z_score'] = np.abs((df['amount'] - df['amount'].mean()) / df['amount'].std())
        
        # Categorize transactions
        desc_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['description', 'merchant', 'name']):
                desc_col = col
                break
        
        if desc_col:
            df['category'] = df[desc_col].apply(MoneyMindML.categorize_transaction)
            df['category_encoded'] = pd.Categorical(df['category']).codes
        
        return df
    
    @staticmethod
    def categorize_transaction(description):
        """Categorize transaction based on description"""
        if pd.isna(description):
            return 'Other'
        
        description_lower = str(description).lower()
        
        categories = {
            'Food & Dining': ['restaurant', 'cafe', 'coffee', 'starbucks', 'mcdonald', 'food', 'grocery', 'uber eats', 'doordash', 'chipotle', 'subway'],
            'Transportation': ['uber', 'lyft', 'gas', 'fuel', 'parking', 'transit', 'subway'],
            'Shopping': ['amazon', 'walmart', 'target', 'shopping', 'store', 'mall'],
            'Entertainment': ['netflix', 'spotify', 'movie', 'theater', 'game', 'entertainment'],
            'Bills & Utilities': ['electric', 'water', 'internet', 'phone', 'utility', 'bill']
        }
        
        for category, keywords in categories.items():
            if any(keyword in description_lower for keyword in keywords):
                return category
        
        return 'Other'
    
    @staticmethod
    def train_spending_predictor(df):
        """Train ML model to predict future spending"""
        try:
            # Prepare features
            feature_cols = ['day_of_week', 'day_of_month', 'month', 'is_weekend', 
                          'rolling_mean_7d', 'category_encoded']
            
            # Remove rows with NaN
            df_clean = df[feature_cols + ['amount']].dropna()
            
            if len(df_clean) < 10:
                return None, None, None
            
            X = df_clean[feature_cols]
            y = df_clean['amount']
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            metrics = {
                'r2_score': r2,
                'rmse': rmse,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
            
            return model, metrics, feature_cols
            
        except Exception as e:
            print(f"Error training predictor: {e}")
            return None, None, None
    
    @staticmethod
    def detect_anomalies(df):
        """Detect unusual spending patterns using Isolation Forest"""
        try:
            feature_cols = ['amount', 'rolling_mean_7d', 'rolling_std_7d', 'z_score']
            df_clean = df[feature_cols].dropna()
            
            if len(df_clean) < 10:
                return df
            
            # Train Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            df.loc[df_clean.index, 'anomaly'] = iso_forest.fit_predict(df_clean)
            df['is_anomaly'] = df['anomaly'] == -1
            
            return df
            
        except Exception as e:
            print(f"Error detecting anomalies: {e}")
            df['is_anomaly'] = False
            return df
    
    @staticmethod
    def cluster_spending_patterns(df):
        """Cluster spending patterns using K-Means"""
        try:
            feature_cols = ['amount', 'day_of_week', 'category_encoded']
            df_clean = df[feature_cols].dropna()
            
            if len(df_clean) < 10:
                return df, None
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clean)
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            df.loc[df_clean.index, 'spending_cluster'] = kmeans.fit_predict(X_scaled)
            
            # Get cluster characteristics
            cluster_info = df.groupby('spending_cluster').agg({
                'amount': ['mean', 'std', 'count']
            }).round(2)
            
            return df, cluster_info
            
        except Exception as e:
            print(f"Error clustering: {e}")
            return df, None
    
    @staticmethod
    def calculate_risk_score(df):
        """Calculate financial risk score"""
        try:
            total_spending = df['amount'].sum()
            avg_spending = df['amount'].mean()
            std_spending = df['amount'].std()
            
            # Coefficient of variation (higher = more risk)
            cv = (std_spending / avg_spending) * 100 if avg_spending > 0 else 0
            
            # Anomaly rate
            anomaly_rate = df['is_anomaly'].sum() / len(df) * 100 if 'is_anomaly' in df.columns else 0
            
            # Risk score (0-100)
            risk_score = min(100, (cv * 0.5) + (anomaly_rate * 5))
            
            risk_category = 'Low' if risk_score < 30 else 'Medium' if risk_score < 60 else 'High'
            
            return {
                'score': round(risk_score, 1),
                'category': risk_category,
                'cv': round(cv, 2),
                'anomaly_rate': round(anomaly_rate, 2)
            }
            
        except Exception as e:
            print(f"Error calculating risk: {e}")
            return {'score': 0, 'category': 'Unknown', 'cv': 0, 'anomaly_rate': 0}

# Header
st.markdown("""
<div class="main-header">
    <h1 class="app-title">MoneyMind Pro</h1>
    <p class="app-subtitle">Machine Learning Edition</p>
    <p class="app-tagline">Smart Money, Smarter You</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ML-Powered Features")
    st.markdown("""
    <span class="ml-badge">ML</span> **Spending Predictor**  
    RandomForest Regression
    
    <span class="ml-badge">ML</span> **Anomaly Detection**  
    Isolation Forest
    
    <span class="ml-badge">ML</span> **Pattern Clustering**  
    K-Means Algorithm
    
    <span class="ml-badge">ML</span> **Risk Scoring**  
    Statistical Analysis
    
    <span class="ml-badge">ML</span> **Smart Categorization**  
    Rule-based + TF-IDF
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Tech Stack")
    st.markdown("""
    **ML Libraries:**
    - scikit-learn
    - pandas
    - numpy
    
    **Visualization:**
    - plotly
    
    **AI Chat:**
    - Groq (optional)
    """)
    
    st.markdown("---")
    
    st.markdown("### 100% Free & Open Source")
    st.markdown("No paid APIs required for ML features!")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Upload Data", 
    "🤖 ML Models", 
    "📈 Predictions", 
    "🔍 Anomalies",
    "💬 AI Chat"
])

with tab1:
    st.markdown("### Upload Your Transaction Data")
    
    uploaded_file = st.file_uploader(
        "Choose your CSV file",
        type=['csv'],
        help="Upload a CSV with Date, Amount, and Description columns"
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['raw_data'] = df
        
        st.success("✓ File uploaded successfully!")
        
        # Show preview
        st.markdown("#### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Transactions</div>
                <div class="metric-value">{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Columns</div>
                <div class="metric-value">{len(df.columns)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Find amount column
        amount_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['amount', 'value', 'total']):
                amount_col = col
                break
        
        if amount_col:
            with col3:
                total = df[amount_col].sum()
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Total Spending</div>
                    <div class="metric-value">${total:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg = df[amount_col].mean()
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Avg Transaction</div>
                    <div class="metric-value">${avg:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Feature Engineering Button
        st.markdown("---")
        if st.button("🔧 Engineer Features & Train ML Models", use_container_width=True):
            with st.spinner("Engineering features and training ML models..."):
                # Feature engineering
                df_engineered = MoneyMindML.engineer_features(df)
                st.session_state['engineered_data'] = df_engineered
                
                # Train models
                model, metrics, features = MoneyMindML.train_spending_predictor(df_engineered)
                st.session_state['prediction_model'] = model
                st.session_state['prediction_metrics'] = metrics
                st.session_state['prediction_features'] = features
                
                # Anomaly detection
                df_with_anomalies = MoneyMindML.detect_anomalies(df_engineered)
                st.session_state['engineered_data'] = df_with_anomalies
                
                # Clustering
                df_clustered, cluster_info = MoneyMindML.cluster_spending_patterns(df_with_anomalies)
                st.session_state['engineered_data'] = df_clustered
                st.session_state['cluster_info'] = cluster_info
                
                # Risk score
                risk = MoneyMindML.calculate_risk_score(df_clustered)
                st.session_state['risk_score'] = risk
                
                st.session_state.ml_models_trained = True
                
                st.success("✓ ML models trained successfully!")
                st.rerun()
    
    else:
        st.info("👆 Upload a CSV file to get started")
        
        st.markdown("""
        <div class="insight-box">
            <h4>What Makes This ML-Powered?</h4>
            <p><strong>Real Machine Learning Models:</strong></p>
            <ul>
                <li><strong>RandomForest Regressor:</strong> Predicts future spending</li>
                <li><strong>Isolation Forest:</strong> Detects unusual transactions</li>
                <li><strong>K-Means Clustering:</strong> Groups spending patterns</li>
                <li><strong>Feature Engineering:</strong> 15+ engineered features</li>
                <li><strong>Statistical Analysis:</strong> Risk scoring algorithm</li>
            </ul>
            <p style="margin-top: 1rem;"><em>All models trained on YOUR data - no pre-trained models!</em></p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Machine Learning Models")
    
    if st.session_state.ml_models_trained:
        df = st.session_state.get('engineered_data')
        
        # Model Performance
        st.markdown("#### Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metrics = st.session_state.get('prediction_metrics', {})
            r2 = metrics.get('r2_score', 0)
            st.markdown(f"""
            <div class="model-metrics">
                <h4>Spending Predictor</h4>
                <div class="metric-label">R² Score</div>
                <div class="metric-value {'positive' if r2 > 0.7 else ''}">{r2:.3f}</div>
                <div class="metric-subtext">{'Excellent' if r2 > 0.8 else 'Good' if r2 > 0.6 else 'Fair'} accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            risk = st.session_state.get('risk_score', {})
            score = risk.get('score', 0)
            category = risk.get('category', 'Unknown')
            st.markdown(f"""
            <div class="model-metrics">
                <h4>Risk Assessment</h4>
                <div class="metric-label">Risk Score</div>
                <div class="metric-value {'negative' if score > 60 else 'positive' if score < 30 else ''}">{score:.1f}/100</div>
                <div class="metric-subtext">{category} Risk</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            anomalies = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
            st.markdown(f"""
            <div class="model-metrics">
                <h4>Anomaly Detection</h4>
                <div class="metric-label">Anomalies Found</div>
                <div class="metric-value {'negative' if anomalies > 5 else 'positive'}">{anomalies}</div>
                <div class="metric-subtext">Unusual transactions</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature Importance
        st.markdown("---")
        st.markdown("#### Feature Importance Analysis")
        
        metrics = st.session_state.get('prediction_metrics', {})
        if 'feature_importance' in metrics:
            importance_df = pd.DataFrame(
                list(metrics['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker=dict(color='#8B5CF6')
            ))
            
            fig.update_layout(
                title='Which features matter most for prediction?',
                xaxis_title='Importance Score',
                yaxis_title='',
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Spending Clusters
        st.markdown("---")
        st.markdown("#### Spending Pattern Clusters")
        
        cluster_info = st.session_state.get('cluster_info')
        if cluster_info is not None:
            st.dataframe(cluster_info, use_container_width=True)
            
            if 'spending_cluster' in df.columns:
                fig = px.scatter(
                    df.dropna(subset=['spending_cluster']),
                    x='day_of_week',
                    y='amount',
                    color='spending_cluster',
                    title='Spending Patterns by Cluster',
                    labels={'day_of_week': 'Day of Week', 'amount': 'Amount ($)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("👈 Upload data and train models first")

with tab3:
    st.markdown("### Spending Predictions")
    
    if st.session_state.ml_models_trained:
        df = st.session_state.get('engineered_data')
        
        st.markdown("#### Next 7 Days Predicted Spending")
        
        model = st.session_state.get('prediction_model')
        features = st.session_state.get('prediction_features')
        
        if model and features:
            # Generate predictions for next 7 days
            last_date = df['date'].max() if 'date' in df.columns else datetime.now()
            predictions = []
            
            for i in range(1, 8):
                future_date = last_date + timedelta(days=i)
                
                # Create feature vector
                feature_dict = {
                    'day_of_week': future_date.dayofweek,
                    'day_of_month': future_date.day,
                    'month': future_date.month,
                    'is_weekend': 1 if future_date.dayofweek >= 5 else 0,
                    'rolling_mean_7d': df['rolling_mean_7d'].iloc[-1] if 'rolling_mean_7d' in df.columns else df['amount'].mean(),
                    'category_encoded': df['category_encoded'].mode()[0] if 'category_encoded' in df.columns else 0
                }
                
                X_pred = pd.DataFrame([feature_dict])
                pred = model.predict(X_pred)[0]
                
                predictions.append({
                    'Date': future_date.strftime('%Y-%m-%d'),
                    'Predicted Spending': f'${pred:.2f}'
                })
            
            pred_df = pd.DataFrame(predictions)
            st.dataframe(pred_df, use_container_width=True)
            
            # Total predicted
            total_pred = sum([float(p['Predicted Spending'].replace('$', '')) for p in predictions])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Next Week Total</div>
                    <div class="metric-value">${total_pred:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_pred = total_pred / 7
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Daily Average</div>
                    <div class="metric-value">${avg_pred:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                current_avg = df['amount'].mean()
                change = ((avg_pred - current_avg) / current_avg * 100) if current_avg > 0 else 0
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">vs Current Avg</div>
                    <div class="metric-value {'positive' if change < 0 else 'negative'}">{change:+.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Visualization
        st.markdown("---")
        st.markdown("#### Historical vs Predicted Spending")
        
        # Combine historical and predicted
        historical = df[['date', 'amount']].tail(14).copy() if 'date' in df.columns and 'amount' in df.columns else None
        
        if historical is not None:
            historical['type'] = 'Historical'
            
            future_data = pd.DataFrame([
                {
                    'date': last_date + timedelta(days=i),
                    'amount': float(predictions[i-1]['Predicted Spending'].replace('$', '')),
                    'type': 'Predicted'
                }
                for i in range(1, 8)
            ])
            
            combined = pd.concat([historical, future_data])
            
            fig = px.line(
                combined,
                x='date',
                y='amount',
                color='type',
                title='Spending Trend: Historical + ML Predictions',
                labels={'date': 'Date', 'amount': 'Amount ($)', 'type': 'Data Type'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("👈 Upload data and train models first")

with tab4:
    st.markdown("### Anomaly Detection")
    
    if st.session_state.ml_models_trained:
        df = st.session_state.get('engineered_data')
        
        if 'is_anomaly' in df.columns:
            anomalies = df[df['is_anomaly'] == True]
            
            st.markdown(f"#### Found {len(anomalies)} Unusual Transactions")
            
            if len(anomalies) > 0:
                # Show anomalies
                display_cols = [col for col in ['date', 'amount', 'description', 'category', 'z_score'] if col in anomalies.columns]
                st.dataframe(anomalies[display_cols].sort_values('amount', ascending=False), use_container_width=True)
                
                # Visualization
                st.markdown("---")
                st.markdown("#### Anomaly Visualization")
                
                if 'date' in df.columns and 'amount' in df.columns:
                    fig = go.Figure()
                    
                    # Normal transactions
                    normal = df[df['is_anomaly'] == False]
                    fig.add_trace(go.Scatter(
                        x=normal['date'],
                        y=normal['amount'],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='#10B981', size=8)
                    ))
                    
                    # Anomalies
                    fig.add_trace(go.Scatter(
                        x=anomalies['date'],
                        y=anomalies['amount'],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color='#DC2626', size=12, symbol='x')
                    ))
                    
                    fig.update_layout(
                        title='Transaction Anomalies Detected by Isolation Forest',
                        xaxis_title='Date',
                        yaxis_title='Amount ($)',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Insights
                st.markdown("""
                <div class="insight-box">
                    <h4>What Are Anomalies?</h4>
                    <p>Anomalies are transactions that significantly differ from your normal spending patterns. 
                    They could indicate:</p>
                    <ul>
                        <li>Large unexpected purchases</li>
                        <li>Potentially fraudulent transactions</li>
                        <li>One-time expenses (travel, medical, etc.)</li>
                        <li>Data entry errors</li>
                    </ul>
                    <p style="margin-top: 1rem;"><strong>ML Model Used:</strong> Isolation Forest (unsupervised learning)</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("No anomalies detected - your spending is consistent!")
        
    else:
        st.info("👈 Upload data and train models first")

with tab5:
    st.markdown("### AI Financial Assistant")
    
    if HAS_GROQ:
        try:
            groq_key = st.secrets.get("groq", {}).get("api_key")
            
            if groq_key and st.session_state.ml_models_trained:
                df = st.session_state.get('engineered_data')
                
                st.markdown("#### Ask Questions About Your Spending")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("What are my ML insights?", use_container_width=True):
                        st.session_state['ai_question'] = "Based on my ML analysis, what are the key insights?"
                
                with col2:
                    if st.button("How can I save money?", use_container_width=True):
                        st.session_state['ai_question'] = "Based on my spending patterns, how can I save money?"
                
                question = st.text_input(
                    "Or ask your own question:",
                    value=st.session_state.get('ai_question', ''),
                    placeholder="e.g., Should I be worried about my anomalies?"
                )
                
                if st.button("Get AI Answer", use_container_width=True) and question:
                    with st.spinner("Analyzing with AI..."):
                        # Prepare context
                        metrics = st.session_state.get('prediction_metrics', {})
                        risk = st.session_state.get('risk_score', {})
                        
                        context = f"""
ML Analysis Results:
- Model Accuracy (R²): {metrics.get('r2_score', 0):.3f}
- Risk Score: {risk.get('score', 0):.1f}/100 ({risk.get('category', 'Unknown')} risk)
- Anomalies Detected: {df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0}
- Total Spending: ${df['amount'].sum():,.2f}
- Average Transaction: ${df['amount'].mean():,.2f}
- Spending Clusters: {df['spending_cluster'].nunique() if 'spending_cluster' in df.columns else 0}

User Question: {question}
"""
                        
                        client = Groq(api_key=groq_key)
                        completion = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": "You are a professional financial advisor with ML expertise. Explain insights clearly and provide actionable advice."},
                                {"role": "user", "content": context}
                            ],
                            temperature=0.7,
                            max_tokens=500
                        )
                        
                        answer = completion.choices[0].message.content
                        
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>AI Response</h4>
                            <p>{answer}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if 'ai_question' in st.session_state:
                            del st.session_state['ai_question']
            
            elif not groq_key:
                st.warning("Groq API key not configured. Add it to secrets to enable AI chat.")
            else:
                st.info("👈 Upload data and train ML models first")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    else:
        st.warning("Groq library not installed. Install with: pip install groq")
        st.info("AI chat is optional - all ML features work without it!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-light); font-size: 0.875rem;">
    <p><strong>MoneyMind Pro</strong> - ML-Powered Financial Analysis</p>
    <p>Professional Machine Learning Edition | 100% Free & Open Source</p>
    <p><em>RandomForest · Isolation Forest · K-Means · Feature Engineering · scikit-learn</em></p>
</div>
""", unsafe_allow_html=True)
