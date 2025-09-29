import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import pickle
import joblib

# Set page config
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 0.25rem solid #1f77b4;
}
.prediction-box {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.prediction-box h3 {
    margin-top: 0;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
}
.stTabs [data-baseweb="tab"] {
    height: 4rem;
}
</style>
""", unsafe_allow_html=True)

def load_models():
    """Load trained models and their associated data"""
    try:
        models = {}
        model_names = ['svm', 'naive_bayes', 'random_forest']
        
        for name in model_names:
            with open(f'models/{name}.pkl', 'rb') as f:
                models[name.replace('_', ' ').title()] = pickle.load(f)
        
        # Load encoders and scaler
        encoders = joblib.load('models/encoders.joblib')
        scaler = joblib.load('models/scaler.joblib')
        
        return models, encoders, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run train_models.ipynb first to train and save the models.")
        return None, None, None

@st.cache_data
def load_and_preprocess_data():
    """Load the TSV data and preprocess it"""
    try:
        columns = [
            'filename', 'label', 'statement', 'subject', 'speaker', 
            'job_title', 'state', 'party', 'true_counts', 'false_counts',
            'half_true_counts', 'barely_true_counts', 'pants_fire_counts', 'context'
        ]
        
        try:
            df = pd.read_csv('train.tsv', sep='\t', header=None, names=columns)
        except FileNotFoundError:
            st.error("train.tsv not found. Please make sure the file exists.")
            return None
        
        return clean_data(df)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean and preprocess the dataframe"""
    df = df.dropna(subset=['label', 'statement', 'subject', 'party'])
    
    numeric_columns = ['true_counts', 'false_counts', 'half_true_counts', 
                      'barely_true_counts', 'pants_fire_counts']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['statement_length'] = df['statement'].str.len()
    df['total_fact_checks'] = (df['true_counts'] + df['false_counts'] + 
                              df['half_true_counts'] + df['barely_true_counts'] + 
                              df['pants_fire_counts'])
    df['subject_category'] = df['subject'].apply(categorize_subject)
    
    # Clean party column
    df['party'] = df['party'].fillna('none')
    df['party'] = df['party'].astype(str).str.lower().str.strip()
    df['party'] = df['party'].replace('', 'none')
    
    df['truthfulness_ratio'] = df['true_counts'] / (df['total_fact_checks'] + 1)
    
    return df

@st.cache_data
def categorize_subject(subject):
    """Categorize subjects into broader categories"""
    if pd.isna(subject) or subject == '' or str(subject).lower() == 'nan':
        return 'other'
    
    subject = str(subject).lower()
    
    if any(word in subject for word in ['health', 'medicare', 'medicaid']):
        return 'healthcare'
    elif any(word in subject for word in ['economy', 'jobs', 'unemployment', 'taxes', 'budget']):
        return 'economy'
    elif any(word in subject for word in ['education', 'school', 'student']):
        return 'education'
    elif any(word in subject for word in ['immigration', 'border']):
        return 'immigration'
    elif any(word in subject for word in ['military', 'war', 'veteran']):
        return 'military'
    elif any(word in subject for word in ['environment', 'climate', 'energy']):
        return 'environment'
    elif any(word in subject for word in ['campaign', 'election', 'voting']):
        return 'political_process'
    else:
        return 'other'

def preprocess_features(df):
    """Preprocess features for machine learning"""
    encoders = {}
    df_processed = df.copy()
    
    # Encode categorical variables
    for col, prefix in [('label', 'label'), ('subject_category', 'subject'), 
                       ('party', 'party'), ('state', 'state'), ('job_title', 'job_title')]:
        le = LabelEncoder()
        df_processed[f'{col}_encoded'] = le.fit_transform(df[col])
        encoders[prefix] = le
    
    feature_cols = [
        'subject_category_encoded', 'party_encoded', 'state_encoded', 'job_title_encoded',
        'true_counts', 'false_counts', 'half_true_counts', 
        'barely_true_counts', 'pants_fire_counts', 'total_fact_checks',
        'statement_length', 'truthfulness_ratio'
    ]
    
    X = df_processed[feature_cols]
    y = df_processed['label_encoded']
    
    return X, y, encoders, df_processed

def initialize_app():
    """Initialize the application by loading models and data"""
    if 'initialized' not in st.session_state:
        with st.spinner("Initializing application..."):
            # Load models
            models, model_encoders, model_scaler = load_models()
            if models is None:
                st.error("Failed to load models. Please run train_models.ipynb first.")
                st.stop()
            
            # Load and preprocess data
            df = load_and_preprocess_data()
            if df is None:
                st.error("Failed to load data. Please check your data source.")
                st.stop()
            
            # Preprocess features
            X, y, encoders, df_processed = preprocess_features(df)
            
            # Store everything in session state
            st.session_state.update({
                'models': models,
                'encoders': model_encoders,
                'scaler': model_scaler,
                'data': df,
                'X': X,
                'y': y,
                'initialized': True
            })
    
    return st.session_state.data

def quick_analysis_tab():
    """Statement analysis interface"""
    st.header("💬 Statement Analysis")
    
    # Text input for statement
    statement_text = st.text_area(
        "Enter the political statement to analyze:",
        height=150,
        placeholder="Enter the political statement here..."
    )
    
    if st.button("Analyze Statement", type="primary"):
        if not statement_text:
            st.error("Please enter a statement to analyze.")
            return
        
        # Use default/average values for other features
        df = st.session_state.data
        input_data = {
            'subject_category': df['subject_category'].mode()[0],
            'party': df['party'].mode()[0],
            'state': df['state'].mode()[0],
            'job_title': df['job_title'].mode()[0],
            'true_counts': int(df['true_counts'].mean()),
            'false_counts': int(df['false_counts'].mean()),
            'half_true_counts': int(df['half_true_counts'].mean()),
            'barely_true_counts': int(df['barely_true_counts'].mean()),
            'pants_fire_counts': int(df['pants_fire_counts'].mean())
        }
        
        make_prediction(statement_text, input_data)

def detailed_analysis_tab():
    """Detailed analysis interface with all features"""
    st.header("🎯 Detailed Analysis")
    
    with st.form("detailed_analysis_form"):
        # Text input for statement
        statement_text = st.text_area(
            "Enter the political statement to analyze:",
            height=150,
            placeholder="Enter the political statement here..."
        )
        
        # Feature inputs in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Speaker Information")
            df = st.session_state.data
            subject_input = st.selectbox("Subject Category", df['subject_category'].unique())
            party_input = st.selectbox("Political Party", df['party'].unique())
            state_input = st.selectbox("State", df['state'].unique())
            job_title_input = st.selectbox("Job Title", df['job_title'].unique())
        
        with col2:
            st.subheader("Historical Fact Checks")
            true_counts = st.number_input("Previous True Statements", 0, 100, 10)
            false_counts = st.number_input("Previous False Statements", 0, 100, 10)
            half_true_counts = st.number_input("Half-True Statements", 0, 50, 5)
            barely_true_counts = st.number_input("Barely-True Statements", 0, 50, 5)
            pants_fire_counts = st.number_input("Pants-on-Fire Statements", 0, 30, 2)
        
        submitted = st.form_submit_button("Analyze Statement", type="primary")
        
        if submitted:
            if not statement_text:
                st.error("Please enter a statement to analyze.")
                return
            
            input_data = {
                'subject_category': subject_input,
                'party': party_input,
                'state': state_input,
                'job_title': job_title_input,
                'true_counts': true_counts,
                'false_counts': false_counts,
                'half_true_counts': half_true_counts,
                'barely_true_counts': barely_true_counts,
                'pants_fire_counts': pants_fire_counts
            }
            
            make_prediction(statement_text, input_data)

def data_overview_tab():
    """Data overview and model performance interface"""
    st.header("📊 Data Overview")
    
    df = st.session_state.data
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Unique Labels", df['label'].nunique())
    with col3:
        st.metric("Unique Subjects", df['subject'].nunique())
    with col4:
        st.metric("Unique Parties", df['party'].nunique())
    
    # Label distribution
    st.subheader("Truth Label Distribution")
    label_counts = df['label'].value_counts()
    fig = px.pie(
        values=label_counts.values,
        names=label_counts.index,
        title="Distribution of Truth Labels",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Subject distribution
    st.subheader("Subject Categories")
    subject_counts = df['subject_category'].value_counts()
    fig = px.bar(
        x=subject_counts.index,
        y=subject_counts.values,
        title="Distribution of Subject Categories",
        labels={'x': 'Category', 'y': 'Count'},
        color=subject_counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

def make_prediction(statement_text, input_data):
    """Make predictions using all models"""
    try:
        # Calculate derived features
        statement_length = len(statement_text)
        total_fact_checks = sum([
            input_data['true_counts'],
            input_data['false_counts'],
            input_data['half_true_counts'],
            input_data['barely_true_counts'],
            input_data['pants_fire_counts']
        ])
        truthfulness_ratio = input_data['true_counts'] / (total_fact_checks + 1)
        
        # Encode categorical inputs
        encoders = st.session_state.encoders
        encoded_inputs = {
            'subject': encoders['subject'].transform([input_data['subject_category']])[0],
            'party': encoders['party'].transform([input_data['party']])[0],
            'state': encoders['state'].transform([input_data['state']])[0],
            'job_title': encoders['job_title'].transform([input_data['job_title']])[0]
        }
        
        # Prepare input data
        input_features = np.array([[
            encoded_inputs['subject'],
            encoded_inputs['party'],
            encoded_inputs['state'],
            encoded_inputs['job_title'],
            input_data['true_counts'],
            input_data['false_counts'],
            input_data['half_true_counts'],
            input_data['barely_true_counts'],
            input_data['pants_fire_counts'],
            total_fact_checks,
            statement_length,
            truthfulness_ratio
        ]])
        
        # Display the input statement
        st.markdown("### Statement to Analyze")
        st.markdown(f"> _{statement_text}_")
        
        # Make predictions with each model
        for model_name, model in st.session_state.models.items():
            st.markdown(f"### {model_name} Analysis")
            
            pred_class = model.predict(input_features)[0]
            probabilities = model.predict_proba(input_features)[0]
            confidence = f"(Confidence: {probabilities[pred_class]:.3f})"
            
            predicted_label = encoders['label'].inverse_transform([pred_class])[0]
            
            # Display prediction with color coding
            label_colors = {
                'true': ['#c6efce', '#006100'],
                'mostly-true': ['#c6efce', '#006100'],
                'half-true': ['#fff2cc', '#9c6500'],
                'barely-true': ['#fff2cc', '#9c6500'],
                'false': ['#ffc7ce', '#9c0006'],
                'pants-fire': ['#ffc7ce', '#9c0006']
            }
            
            colors = label_colors.get(predicted_label.lower(), ['#b4c6e7', '#000000'])
            
            st.markdown(
                f"""<div style='padding: 1.5rem; border-radius: 0.5rem; 
                background-color: {colors[0]}; margin-bottom: 1rem;'>
                <h3 style='margin:0; color: {colors[1]}; font-size: 1.5rem;'>
                Verdict: {predicted_label.upper()}</h3>
                <p style='margin:0; color: {colors[1]}; font-size: 1.1rem;'>{confidence}</p>
                </div>""",
                unsafe_allow_html=True
            )
            
            # Show probability distribution
            proba_df = pd.DataFrame({
                'Label': encoders['label'].classes_,
                'Probability': probabilities,
                'Percentage': probabilities * 100
            }).sort_values('Probability', ascending=False)
            
            fig = px.bar(
                proba_df,
                x='Label',
                y='Percentage',
                title=f"Confidence Distribution - {model_name}",
                labels={'Percentage': 'Confidence (%)'},
                color='Percentage',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(xaxis_tickangle=45, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")

def main():
    st.markdown('<h1 class="main-header">Fake News Detection</h1>', unsafe_allow_html=True)
    
    # Initialize application (loads models and data)
    initialize_app()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs([
        "💬 Analysis",
        "🎯 Detailed Analysis",
        "📊 Data Overview"
    ])
    
    with tab1:
        quick_analysis_tab()
    
    with tab2:
        detailed_analysis_tab()
    
    with tab3:
        data_overview_tab()

if __name__ == "__main__":
    main()