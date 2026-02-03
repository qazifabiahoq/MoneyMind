import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from groq import Groq
    HAS_GROQ = True
except:
    HAS_GROQ = False

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except:
    HAS_OCR = False

# Page Config
st.set_page_config(
    page_title="Prism",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Banking CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-blue: #003D82;
        --accent-blue: #0066CC;
        --success-green: #00C389;
        --bg-light: #F5F7FA;
        --bg-white: #FFFFFF;
        --text-dark: #1A1A1A;
        --text-medium: #5E6C84;
        --text-light: #8993A4;
        --border: #DFE1E6;
        --shadow: 0 2px 8px rgba(9, 30, 66, 0.08);
        --shadow-lg: 0 8px 24px rgba(9, 30, 66, 0.12);
    }
    
    html {
        scroll-behavior: smooth;
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background: var(--bg-light);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Hide Sidebar Completely */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #001B3D 0%, #003D82 40%, #0052A3 70%, #0066CC 100%);
        padding: 5rem 2rem 4rem 2rem;
        margin: -6rem -5rem 3rem -5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .main-header::after {
        content: "";
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(0, 195, 137, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .header-content {
        position: relative;
        z-index: 10;
        text-align: center;
    }
    
    .app-title {
        font-size: 4.5rem;
        font-weight: 900;
        color: #FFFFFF !important;
        margin: 0;
        letter-spacing: 4px;
        text-shadow: 0 0 30px rgba(255, 255, 255, 1), 0 4px 25px rgba(255, 255, 255, 0.8), 0 8px 40px rgba(0, 0, 0, 0.5);
        text-transform: uppercase;
        -webkit-text-fill-color: #FFFFFF !important;
        paint-order: stroke fill;
        filter: brightness(1.2) contrast(1.3);
    }
    
    .app-tagline {
        font-size: 1.5rem;
        color: #FFFFFF !important;
        margin-top: 1.25rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.9), 0 2px 12px rgba(255, 255, 255, 0.7);
        opacity: 1;
        -webkit-text-fill-color: #FFFFFF !important;
        filter: brightness(1.15);
    }
    
    /* Tabs - Professional Banking Style - STICKY */
    .stTabs {
        background: var(--bg-white);
        border-radius: 8px 8px 0 0;
        margin-top: 1rem;
        box-shadow: var(--shadow);
        position: sticky;
        top: 0;
        z-index: 999;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-white);
        border-bottom: 2px solid var(--border);
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: transparent;
        border: none;
        border-bottom: 3px solid transparent;
        border-radius: 0;
        padding: 0 2.5rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-medium) !important;
        transition: all 0.2s ease;
        letter-spacing: 0.3px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #F0F4F8;
        color: var(--primary-blue) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-blue) !important;
        color: #FFFFFF !important;
        border-bottom-color: var(--primary-blue);
    }
    
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] div {
        color: #FFFFFF !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding: 2rem 1rem;
        background: var(--bg-white);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-dark) !important;
        font-weight: 700;
    }
    
    h1 { font-size: 2.5rem; }
    h2 { font-size: 2rem; }
    h3 { 
        font-size: 1.5rem; 
        margin-bottom: 1.5rem; 
        color: var(--text-dark) !important;
        font-weight: 700;
    }
    h4 { font-size: 1.25rem; }
    
    p, div, span, label {
        color: var(--text-medium);
        line-height: 1.6;
    }
    
    /* Make all labels more visible */
    label {
        color: var(--text-dark) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Improve radio button labels */
    .stRadio label {
        color: var(--text-dark) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stRadio > div {
        color: var(--text-dark) !important;
    }
    
    .stRadio > label {
        color: var(--text-dark) !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    
    /* Text inputs and areas */
    .stTextInput label,
    .stTextArea label,
    .stSelectbox label {
        color: var(--text-dark) !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    
    /* Cards */
    .metric-card {
        background: var(--bg-white);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(9, 30, 66, 0.1);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 8px 24px rgba(9, 30, 66, 0.15);
        border-color: var(--accent-blue);
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 700;
        color: var(--text-light);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.75rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary-blue);
        margin: 0.75rem 0;
        line-height: 1.2;
    }
    
    .metric-value.positive {
        color: var(--success-green);
    }
    
    .metric-value.negative {
        color: #FF5630;
    }
    
    .metric-description {
        font-size: 0.875rem;
        color: var(--text-medium);
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Buttons - MAXIMUM CONTRAST */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--accent-blue) 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px;
        padding: 1rem 2.5rem;
        font-size: 1.1rem !important;
        font-weight: 900 !important;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        height: 56px;
        box-shadow: 0 4px 12px rgba(0, 61, 130, 0.25);
        letter-spacing: 0.8px;
    }
    
    .stButton > button *,
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #FFFFFF !important;
        font-weight: 900 !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0052A3 0%, #0077E6 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 61, 130, 0.35);
        color: #FFFFFF !important;
    }
    
    .stButton > button:hover *,
    .stButton > button:hover p,
    .stButton > button:hover span,
    .stButton > button:hover div {
        color: #FFFFFF !important;
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* File Uploader - WHITE BACKGROUND, DARK TEXT */
    [data-testid="stFileUploader"] {
        background: #FFFFFF !important;
        border: 3px dashed var(--accent-blue) !important;
        border-radius: 12px;
        padding: 3rem 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-blue);
        background: #F8FBFF !important;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.15);
    }
    
    [data-testid="stFileUploader"] section {
        border: none;
        padding: 1rem;
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: var(--primary-blue) !important;
        color: #FFFFFF !important;
        border-radius: 6px;
        font-weight: 700 !important;
    }
    
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] small {
        color: #1A1A1A !important;
        font-weight: 600 !important;
        background: transparent !important;
    }
    
    /* Force ALL file uploader children to have dark text on white */
    [data-testid="stFileUploader"] *:not(button) {
        color: #1A1A1A !important;
        background: transparent !important;
    }
    
    /* File uploader drag area */
    [data-testid="stFileUploader"] > div {
        background: #FFFFFF !important;
    }
    
    /* Info Boxes */
    .info-card {
        background: linear-gradient(135deg, #F0F7FF 0%, #E6F2FF 100%);
        border-left: 4px solid var(--primary-blue);
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
    }
    
    .info-card h4 {
        color: var(--primary-blue);
        margin-bottom: 0.75rem;
        font-size: 1.125rem;
    }
    
    .info-card p {
        color: var(--text-dark);
        margin: 0.5rem 0;
    }
    
    /* Success/Warning/Info Messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 8px !important;
        padding: 1rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #E6F9F0 0%, #D1F4E0 100%) !important;
        border-left: 4px solid var(--success-green) !important;
        color: #00613D !important;
    }
    
    .stSuccess p, .stSuccess div, .stSuccess span {
        color: #00613D !important;
        font-weight: 600 !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #E6F2FF 0%, #D6EBFF 100%) !important;
        border-left: 4px solid var(--accent-blue) !important;
        color: #002855 !important;
    }
    
    .stInfo p, .stInfo div, .stInfo span {
        color: #002855 !important;
        font-weight: 600 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #FFF4E6 0%, #FFE8CC 100%) !important;
        border-left: 4px solid #FF991F !important;
        color: #663C00 !important;
    }
    
    .stWarning p, .stWarning div, .stWarning span {
        color: #663C00 !important;
        font-weight: 600 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #FFE6E6 0%, #FFD6D6 100%) !important;
        border-left: 4px solid #FF5630 !important;
        color: #6B0000 !important;
    }
    
    .stError p, .stError div, .stError span {
        color: #6B0000 !important;
        font-weight: 600 !important;
    }
    
    /* Input Fields - MAXIMUM CONTRAST */
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox select,
    input[type="text"],
    input[type="email"],
    input[type="number"],
    textarea {
        background: var(--bg-white) !important;
        border: 2px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 0.875rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.2s ease;
        color: #1A1A1A !important;
        font-weight: 500 !important;
    }
    
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder,
    input::placeholder,
    textarea::placeholder {
        color: #8993A4 !important;
        font-weight: 400 !important;
    }
    
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    input:focus,
    textarea:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1) !important;
        color: #1A1A1A !important;
    }
    
    /* Force input text to be dark - AGGRESSIVE */
    input[class*="st-"],
    textarea[class*="st-"],
    .stTextInput input,
    .stTextArea textarea,
    div[data-baseweb="input"] input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    .stTextInput div[data-baseweb="input"] {
        background: white !important;
    }
    
    .stTextInput div[data-baseweb="input"] input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Data Tables */
    .dataframe {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-white);
        border: 1px solid var(--border);
        border-radius: 8px;
        font-weight: 600;
        color: var(--text-dark) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--light-blue);
    }
    
    .streamlit-expanderContent {
        background: var(--bg-white);
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 8px 8px;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 3rem 0 2rem 0;
        margin-top: 4rem;
        border-top: 1px solid var(--border);
        color: var(--text-light);
        font-size: 0.875rem;
    }
    
    .app-footer strong {
        color: var(--text-dark);
        font-weight: 600;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-light);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-light);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0  # 0=Upload, 1=Dashboard, 2=Forecast, 3=Alerts, 4=Assistant

# ML Engine Class
class FinancialIntelligence:
    """Backend ML engine - invisible to users"""
    
    @staticmethod
    def process_data(df):
        """Feature engineering"""
        df = df.copy()
        
        # Find date column
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
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Find amount column
        amount_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['amount', 'value', 'total', 'price']):
                amount_col = col
                break
        
        if amount_col:
            df['amount'] = pd.to_numeric(df[amount_col], errors='coerce')
            df['amount'] = df['amount'].abs()
            
            # Rolling statistics
            df['rolling_mean_7d'] = df['amount'].rolling(window=7, min_periods=1).mean()
            df['rolling_std_7d'] = df['amount'].rolling(window=7, min_periods=1).std()
            df['rolling_max_7d'] = df['amount'].rolling(window=7, min_periods=1).max()
            
            # Z-score
            mean_amount = df['amount'].mean()
            std_amount = df['amount'].std()
            df['z_score'] = (df['amount'] - mean_amount) / std_amount if std_amount > 0 else 0
            
            # Spending velocity
            df['spending_velocity'] = df['amount'].diff().fillna(0)
            df['cumulative_spending'] = df['amount'].cumsum()
        
        # Auto-categorize
        df['category'] = 'Other'
        if 'description' in df.columns or 'Description' in df.columns:
            desc_col = 'description' if 'description' in df.columns else 'Description'
            df['category'] = df[desc_col].apply(FinancialIntelligence.categorize_transaction)
        
        # Encode category
        category_map = {'Food': 0, 'Transportation': 1, 'Shopping': 2, 'Bills': 3, 'Entertainment': 4, 'Other': 5}
        df['category_encoded'] = df['category'].map(category_map).fillna(5)
        
        return df
    
    @staticmethod
    def categorize_transaction(description):
        """Smart categorization"""
        if pd.isna(description):
            return 'Other'
        
        desc = str(description).lower()
        
        food_keywords = ['restaurant', 'cafe', 'coffee', 'food', 'grocery', 'starbucks', 'mcdonald', 'pizza']
        transport_keywords = ['uber', 'lyft', 'gas', 'fuel', 'parking', 'transit', 'bus', 'taxi']
        shopping_keywords = ['amazon', 'store', 'shop', 'retail', 'clothing', 'mall']
        bills_keywords = ['utility', 'electric', 'water', 'rent', 'mortgage', 'insurance', 'phone', 'internet']
        entertainment_keywords = ['movie', 'theater', 'game', 'spotify', 'netflix', 'concert', 'gym']
        
        if any(kw in desc for kw in food_keywords):
            return 'Food'
        elif any(kw in desc for kw in transport_keywords):
            return 'Transportation'
        elif any(kw in desc for kw in shopping_keywords):
            return 'Shopping'
        elif any(kw in desc for kw in bills_keywords):
            return 'Bills'
        elif any(kw in desc for kw in entertainment_keywords):
            return 'Entertainment'
        else:
            return 'Other'
    
    @staticmethod
    def build_forecast_model(df):
        """Train spending predictor"""
        try:
            features = ['day_of_week', 'day_of_month', 'month', 'is_weekend', 'rolling_mean_7d', 'category_encoded']
            features = [f for f in features if f in df.columns]
            
            X = df[features].fillna(0)
            y = df['amount']
            
            if len(X) < 20:
                return None, None, None
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            feature_importance = dict(zip(features, model.feature_importances_))
            
            return model, {'r2': r2, 'rmse': rmse, 'importance': feature_importance}, features
            
        except Exception as e:
            print(f"Model error: {e}")
            return None, None, None
    
    @staticmethod
    def detect_unusual_activity(df):
        """Anomaly detection"""
        try:
            features = ['amount', 'rolling_mean_7d', 'rolling_std_7d', 'z_score']
            features = [f for f in features if f in df.columns]
            
            X = df[features].fillna(0)
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            df['is_anomaly'] = iso_forest.fit_predict(X) == -1
            
            return df
            
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            df['is_anomaly'] = False
            return df
    
    @staticmethod
    def discover_patterns(df):
        """Clustering"""
        try:
            features = ['amount', 'day_of_week', 'category_encoded']
            features = [f for f in features if f in df.columns]
            
            X = df[features].fillna(0)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            df['spending_pattern'] = kmeans.fit_predict(X_scaled)
            
            # Describe patterns
            pattern_summary = df.groupby('spending_pattern').agg({
                'amount': ['mean', 'count'],
                'category': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
            }).round(2)
            
            return df, pattern_summary
            
        except Exception as e:
            print(f"Clustering error: {e}")
            return df, None
    
    @staticmethod
    def calculate_health_score(df):
        """Financial wellness scoring"""
        try:
            avg_spending = df['amount'].mean()
            std_spending = df['amount'].std()
            
            # Coefficient of variation
            cv = (std_spending / avg_spending) * 100 if avg_spending > 0 else 0
            
            # Anomaly rate
            anomaly_rate = df['is_anomaly'].sum() / len(df) * 100 if 'is_anomaly' in df.columns else 0
            
            # Risk score (inverted for wellness score)
            risk_score = min(100, (cv * 0.5) + (anomaly_rate * 5))
            wellness_score = max(0, 100 - risk_score)
            
            if wellness_score >= 70:
                category = 'Excellent'
            elif wellness_score >= 50:
                category = 'Good'
            elif wellness_score >= 30:
                category = 'Fair'
            else:
                category = 'Needs Attention'
            
            return {
                'score': round(wellness_score, 0),
                'category': category,
                'consistency': round(100 - cv, 1),
                'unusual_rate': round(anomaly_rate, 1)
            }
            
        except Exception as e:
            print(f"Wellness score error: {e}")
            return {'score': 50, 'category': 'Unknown', 'consistency': 0, 'unusual_rate': 0}
    
    @staticmethod
    def extract_from_image(image_file):
        """OCR to extract transactions from receipt/screenshot"""
        if not HAS_OCR:
            return None
        
        try:
            image = Image.open(image_file)
            
            # OCR
            text = pytesseract.image_to_string(image)
            
            # Simple parsing (can be enhanced)
            lines = text.split('\n')
            transactions = []
            
            for line in lines:
                # Look for amount patterns ($XX.XX or XX.XX)
                import re
                amounts = re.findall(r'\$?(\d+\.\d{2})', line)
                
                if amounts:
                    transactions.append({
                        'Date': datetime.now().strftime('%Y-%m-%d'),
                        'Amount': float(amounts[0]),
                        'Description': line.split(amounts[0])[0].strip()[:50]
                    })
            
            if transactions:
                return pd.DataFrame(transactions)
            else:
                return None
                
        except Exception as e:
            print(f"OCR error: {e}")
            return None

# Header
st.markdown("""
<div class="main-header">
    <div class="header-content">
        <h1 class="app-title">PRISM</h1>
        <p class="app-tagline">Your Financial Truth</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Upload", 
    "Dashboard", 
    "Forecast", 
    "Alerts",
    "Assistant"
])

# TAB 1: UPLOAD
with tab1:
    st.markdown("### Upload Your Transactions")
    st.markdown("Get instant insights from your spending data")
    
    st.markdown("---")
    
    upload_method = st.radio(
        "Choose upload method:",
        ["Spreadsheet (CSV)", "Photo of Receipt"],
        horizontal=True
    )
    
    st.markdown("---")
    
    uploaded_data = None
    
    if upload_method == "Spreadsheet (CSV)":
        uploaded_file = st.file_uploader(
            "Upload your transaction history (CSV)",
            type=['csv'],
            help="CSV file with Date, Amount, and Description columns"
        )
        
        if uploaded_file:
            uploaded_data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully")
    
    else:
        if HAS_OCR:
            uploaded_image = st.file_uploader(
                "Upload a photo of your receipt",
                type=['png', 'jpg', 'jpeg'],
                help="Take a clear photo of your receipt - we'll extract the transactions"
            )
            
            if uploaded_image:
                with st.spinner("Reading your receipt..."):
                    uploaded_data = FinancialIntelligence.extract_from_image(uploaded_image)
                    
                    if uploaded_data is not None:
                        st.success("Receipt processed successfully!")
                    else:
                        st.warning("""
                        **Couldn't read this receipt**
                        
                        For best results:
                        - Make sure the photo is clear and well-lit
                        - Receipt should be flat (not crumpled)
                        - All text should be visible
                        
                        Or try uploading a spreadsheet (CSV) instead - it's faster and more reliable!
                        """)
        else:
            st.info("""
            **Photo Upload Feature**
            
            Receipt scanning is currently being set up.
            
            Please use spreadsheet (CSV) upload - it's fast and works perfectly!
            """)
    
    if uploaded_data is not None:
        st.session_state['raw_data'] = uploaded_data
        
        # Preview
        st.markdown("#### Your Data")
        st.dataframe(uploaded_data.head(10), use_container_width=True)
        
        # Quick Stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Transactions</div>
                <div class="metric-value">{len(uploaded_data):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Data Fields</div>
                <div class="metric-value">{len(uploaded_data.columns)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Find amount column
        amount_col = None
        for col in uploaded_data.columns:
            if any(keyword in col.lower() for keyword in ['amount', 'value', 'total']):
                amount_col = col
                break
        
        if amount_col:
            with col3:
                total = uploaded_data[amount_col].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Spending</div>
                    <div class="metric-value">${total:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg = uploaded_data[amount_col].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average Amount</div>
                    <div class="metric-value">${avg:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Analyze Button
        st.markdown("---")
        
        # Show different button based on state
        if st.session_state.analysis_complete:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                <div style="background: #E6F9F0; padding: 1rem; border-radius: 8px; border-left: 3px solid #00C389;">
                    <p style="color: #00613D; margin: 0; font-weight: 600;">
                    Analysis complete! Click Dashboard tab above to view results.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("Analyze Again", use_container_width=True):
                    # Re-run analysis with existing data
                    with st.spinner("Re-analyzing your financial data..."):
                        df_processed = FinancialIntelligence.process_data(uploaded_data)
                        st.session_state['data'] = df_processed
                        
                        model, metrics, features = FinancialIntelligence.build_forecast_model(df_processed)
                        st.session_state['forecast_model'] = model
                        st.session_state['forecast_metrics'] = metrics
                        st.session_state['forecast_features'] = features
                        
                        df_processed = FinancialIntelligence.detect_unusual_activity(df_processed)
                        st.session_state['data'] = df_processed
                        
                        df_processed, pattern_summary = FinancialIntelligence.discover_patterns(df_processed)
                        st.session_state['data'] = df_processed
                        st.session_state['patterns'] = pattern_summary
                        
                        wellness = FinancialIntelligence.calculate_health_score(df_processed)
                        st.session_state['wellness'] = wellness
                        
                        st.success("Analysis updated!")
        else:
            if st.button("Analyze My Spending", use_container_width=True):
                with st.spinner("Analyzing your financial data..."):
                    # Process data
                    df_processed = FinancialIntelligence.process_data(uploaded_data)
                    st.session_state['data'] = df_processed
                    
                    # Build forecast
                    model, metrics, features = FinancialIntelligence.build_forecast_model(df_processed)
                    st.session_state['forecast_model'] = model
                    st.session_state['forecast_metrics'] = metrics
                    st.session_state['forecast_features'] = features
                    
                    # Detect anomalies
                    df_processed = FinancialIntelligence.detect_unusual_activity(df_processed)
                    st.session_state['data'] = df_processed
                    
                    # Find patterns
                    df_processed, pattern_summary = FinancialIntelligence.discover_patterns(df_processed)
                    st.session_state['data'] = df_processed
                    st.session_state['patterns'] = pattern_summary
                    
                    # Calculate wellness
                    wellness = FinancialIntelligence.calculate_health_score(df_processed)
                    st.session_state['wellness'] = wellness
                    
                    st.session_state.analysis_complete = True
                    
                    st.success("Analysis complete!")
                    
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #00C389 0%, #00A876 100%); 
                                color: white; 
                                padding: 2rem; 
                                border-radius: 12px; 
                                text-align: center;
                                margin: 1.5rem 0;
                                box-shadow: 0 4px 12px rgba(0, 195, 137, 0.3);">
                        <h2 style="color: white; margin: 0 0 1rem 0; font-size: 1.75rem;">Your Financial Analysis is Ready!</h2>
                        <p style="color: white; font-size: 1.1rem; margin: 0 0 1.5rem 0; opacity: 0.95;">
                            Click the <strong>Dashboard</strong> tab above to see your complete financial insights
                        </p>
                        <div style="background: rgba(255,255,255,0.2); 
                                    padding: 1rem; 
                                    border-radius: 8px;
                                    border: 2px solid rgba(255,255,255,0.3);">
                            <p style="color: white; margin: 0; font-size: 0.95rem;">
                                <strong>Look up - click "Dashboard" to view your results</strong>
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.rerun()
    
    else:
        pass  # No file uploaded yet

# TAB 2: DASHBOARD
with tab2:
    st.markdown("### Financial Dashboard")
    
    if st.session_state.analysis_complete:
        df = st.session_state.get('data')
        
        # Show welcome message if just analyzed
        if 'dashboard_viewed' not in st.session_state:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #E6F2FF 0%, #D6EBFF 100%); 
                        padding: 1.5rem; 
                        border-radius: 12px; 
                        border-left: 4px solid #0066CC;
                        margin-bottom: 2rem;">
                <h4 style="color: #003D82; margin: 0 0 0.5rem 0;">Welcome to Your Dashboard!</h4>
                <p style="color: #1A1A1A; margin: 0; font-size: 0.95rem;">
                    Here's your complete financial overview. Scroll down to explore:
                    <strong>Forecast</strong>, <strong>Alerts</strong>, and <strong>AI Assistant</strong> tabs for deeper insights.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.dashboard_viewed = True
        
        # Overview Metrics
        st.markdown("#### Your Financial Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metrics = st.session_state.get('forecast_metrics')
            if metrics and isinstance(metrics, dict):
                r2 = metrics.get('r2', 0)
                confidence = r2 * 100
            else:
                confidence = 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Forecast Confidence</div>
                <div class="metric-value {'positive' if confidence > 70 else ''}">{confidence:.0f}%</div>
                <div class="metric-description">{'Excellent' if confidence > 80 else 'Good' if confidence > 60 else 'Fair'} prediction accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            wellness = st.session_state.get('wellness')
            if wellness and isinstance(wellness, dict):
                score = wellness.get('score', 0)
                category = wellness.get('category', 'Unknown')
            else:
                score = 0
                category = 'Unknown'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Financial Wellness</div>
                <div class="metric-value {'positive' if score >= 70 else 'negative' if score < 30 else ''}">{score:.0f}</div>
                <div class="metric-description">{category}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            anomalies = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Activity Alerts</div>
                <div class="metric-value {'negative' if anomalies > 5 else 'positive'}">{anomalies}</div>
                <div class="metric-description">Unusual transactions</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Spending by Category
        st.markdown("---")
        st.markdown("#### Spending by Category")
        
        if 'category' in df.columns and 'amount' in df.columns:
            category_spending = df.groupby('category')['amount'].sum().sort_values(ascending=False)
            
            fig = px.pie(
                values=category_spending.values,
                names=category_spending.index,
                title='Where Your Money Goes',
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400, margin=dict(t=40, b=20, l=20, r=20))
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Spending Patterns
        st.markdown("---")
        st.markdown("#### Your Spending Patterns")
        
        patterns = st.session_state.get('patterns')
        if patterns is not None:
            st.dataframe(patterns, use_container_width=True)
        
    else:
        st.info("Upload your transactions and analyze to view your dashboard")

# TAB 3: FORECAST
with tab3:
    st.markdown("### 7-Day Spending Forecast")
    
    if st.session_state.analysis_complete:
        df = st.session_state.get('data')
        model = st.session_state.get('forecast_model')
        features = st.session_state.get('forecast_features')
        
        if model and features:
            st.markdown("#### Predicted Spending")
            
            # Generate predictions
            last_date = df['date'].max() if 'date' in df.columns else datetime.now()
            predictions = []
            
            for i in range(1, 8):
                future_date = last_date + timedelta(days=i)
                
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
                    'Date': future_date.strftime('%A, %b %d'),
                    'Predicted Amount': f'${pred:.2f}'
                })
            
            pred_df = pd.DataFrame(predictions)
            st.dataframe(pred_df, use_container_width=True)
            
            # Summary metrics
            total_pred = sum([float(p['Predicted Amount'].replace('$', '')) for p in predictions])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Week Total</div>
                    <div class="metric-value">${total_pred:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_pred = total_pred / 7
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Daily Average</div>
                    <div class="metric-value">${avg_pred:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                current_avg = df['amount'].mean()
                change = ((avg_pred - current_avg) / current_avg * 100) if current_avg > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">vs Current Average</div>
                    <div class="metric-value {'positive' if change < 0 else 'negative'}">{change:+.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization
            st.markdown("---")
            st.markdown("#### Spending Trend")
            
            if 'date' in df.columns and 'amount' in df.columns:
                historical = df[['date', 'amount']].tail(14).copy()
                historical['type'] = 'Historical'
                
                future_data = pd.DataFrame([
                    {
                        'date': last_date + timedelta(days=i),
                        'amount': float(predictions[i-1]['Predicted Amount'].replace('$', '')),
                        'type': 'Forecast'
                    }
                    for i in range(1, 8)
                ])
                
                combined = pd.concat([historical, future_data])
                
                fig = px.line(
                    combined,
                    x='date',
                    y='amount',
                    color='type',
                    title='Historical Spending + 7-Day Forecast',
                    labels={'date': 'Date', 'amount': 'Amount ($)', 'type': ''},
                    color_discrete_map={'Historical': '#003D82', 'Forecast': '#00C389'}
                )
                
                fig.update_layout(height=400, margin=dict(t=40, b=20, l=20, r=20))
                
                st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Upload transactions and analyze to see your forecast")

# TAB 4: ALERTS
with tab4:
    st.markdown("### Activity Alerts")
    
    if st.session_state.analysis_complete:
        df = st.session_state.get('data')
        
        if 'is_anomaly' in df.columns:
            anomalies = df[df['is_anomaly'] == True]
            
            st.markdown(f"#### {len(anomalies)} Unusual Transactions Detected")
            
            if len(anomalies) > 0:
                # Display anomalies
                display_cols = [col for col in ['date', 'amount', 'description', 'category'] if col in anomalies.columns]
                if display_cols:
                    st.dataframe(
                        anomalies[display_cols].sort_values('amount', ascending=False),
                        use_container_width=True
                    )
                
                # Visualization
                st.markdown("---")
                st.markdown("#### Activity Pattern")
                
                if 'date' in df.columns and 'amount' in df.columns:
                    fig = go.Figure()
                    
                    normal = df[df['is_anomaly'] == False]
                    fig.add_trace(go.Scatter(
                        x=normal['date'],
                        y=normal['amount'],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='#003D82', size=8)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=anomalies['date'],
                        y=anomalies['amount'],
                        mode='markers',
                        name='Unusual',
                        marker=dict(color='#FF5630', size=12, symbol='x')
                    ))
                    
                    fig.update_layout(
                        title='Normal vs Unusual Transactions',
                        xaxis_title='Date',
                        yaxis_title='Amount ($)',
                        height=400,
                        margin=dict(t=40, b=20, l=20, r=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("Great news! No unusual activity detected in your transactions.")
        
    else:
        st.info("Upload transactions and analyze to see alerts")

# TAB 5: ASSISTANT
with tab5:
    st.markdown("### Financial Assistant")
    st.markdown("*Powered by Advanced AI (Llama 3.3 70B)*")
    
    if HAS_GROQ:
        try:
            groq_key = st.secrets.get("groq", {}).get("api_key")
            
            if st.session_state.analysis_complete and groq_key:
                df = st.session_state.get('data')
                
                st.markdown("""
                Ask personalized questions about your spending and get AI-powered financial advice based on your actual data.
                """)
                
                question = st.text_input(
                    "Ask me anything about your finances:",
                    placeholder="Example: How can I reduce my spending? What's my biggest expense category?",
                    key="ai_question"
                )
                
                if question:
                    client = Groq(api_key=groq_key)
                    
                    # Build context
                    wellness = st.session_state.get('wellness')
                    wellness_score = wellness.get('score', 0) if wellness and isinstance(wellness, dict) else 0
                    
                    context = f"""
User's financial data summary:
- Total transactions: {len(df)}
- Average spending: ${df['amount'].mean():.2f}
- Wellness score: {wellness_score}/100
- Unusual transactions: {df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0}
- Top spending category: {df['category'].mode()[0] if 'category' in df.columns and len(df['category'].mode()) > 0 else 'Unknown'}

User question: {question}

Provide helpful, actionable financial advice in a friendly, professional tone. Be specific and data-driven.
"""
                    
                    with st.spinner("Generating personalized advice..."):
                        completion = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": context}],
                            max_tokens=500,
                            temperature=0.7
                        )
                        
                        answer = completion.choices[0].message.content
                        
                        st.markdown(f"""
                        <div class="info-card">
                            <h4>AI Financial Advisor</h4>
                            <p>{answer}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            elif not groq_key:
                st.info("""
                **AI Assistant - Generative AI Powered**
                
                This feature uses **Llama 3.3 70B** (state-of-the-art large language model) to provide:
                - Personalized financial advice based on YOUR data
                - Spending optimization recommendations  
                - Budget planning assistance
                - Financial health improvement tips
                
                **Currently being configured for deployment.** All core analysis features work perfectly!
                """)
            else:
                st.info("Analyze your transactions first to unlock AI-powered advice")
        
        except Exception as e:
            st.error(f"Assistant error: {str(e)}")
    
    else:
        st.info("""
        **AI Financial Assistant**
        
        Get personalized advice powered by advanced generative AI (Llama 3.3 70B).
        
        This feature is being configured. All core analysis features work perfectly!
        """)

# Footer
st.markdown("""
<div class="app-footer">
    <p><strong>PRISM</strong></p>
    <p>Your Financial Truth</p>
    <p style="margin-top: 0.5rem;"><em>Secure • Private • Free</em></p>
</div>
""", unsafe_allow_html=True)
