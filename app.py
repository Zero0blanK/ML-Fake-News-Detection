import streamlit as st
import pandas as pd
import pickle
import joblib
import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

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

def clean_text(text):
    """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text):
    """Tokenize and lemmatize text"""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(token) for token in tokens)

def preprocess_statement(statement):
    """Preprocess a single statement"""
    cleaned_text = clean_text(statement)
    return tokenize_and_lemmatize(cleaned_text)

def load_models():
    """Load trained models and their associated data"""
    try:
        # Load only SVM model
        with open(f'models/svm.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load TF-IDF vectorizer
        vectorizer = joblib.load('models/tfidf.joblib')
        
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please run train_models.ipynb first to train and save the models.")
        return None, None

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
    # Drop rows with missing required fields
    df = df.dropna(subset=['label', 'statement'])
    
    # Convert label to boolean if it's not already
    if not df['label'].dtype == bool:
        df['label'] = df['label'].apply(lambda x: str(x).lower() in ['true', 'mostly-true', 'half-true'])
    
    # Clean statement text
    df['processed_statement'] = df['statement'].apply(preprocess_statement)
    
    return df

def initialize_app():
    """Initialize the application by loading models"""
    if 'initialized' not in st.session_state:
        with st.spinner("Initializing application..."):
            # Load model and vectorizer
            model, vectorizer = load_models()
            if model is None:
                st.error("Failed to load model. Please run train_models.ipynb first.")
                st.stop()
            
            # Store in session state
            st.session_state.update({
                'model': model,
                'vectorizer': vectorizer,
                'initialized': True
            })
    
    return True

def detailed_analysis_tab():
    st.header("🎯 Detailed Analysis")
    
    # Create two sub-tabs
    manual_tab, test_tab = st.tabs(["💭 Manual Input", "🔍 Test Examples"])
    
    with manual_tab:
        st.markdown("""
        Enter a statement to analyze using our **Support Vector Machine (SVM)** classifier.

        Support Vector Machine (SVM) finds the optimal boundary to separate true and false statements. It's particularly effective for text classification tasks and high-dimensional data like TF-IDF features.
        """)
        
        # Create a form for input
        with st.form("manual_analysis_form"):
            # Statement input
            statement_text = st.text_area(
                "Enter the political statement to analyze:",
                height=150,
                placeholder="Enter the political statement here..."
            )
            
            # Optional context input
            st.markdown("### Optional Context")
            context = st.text_area(
                "Additional context (if any):",
                height=100,
                placeholder="Enter any additional context about the statement...",
                help="This helps provide background information about the statement."
            )
            
            # Submit button
            submitted = st.form_submit_button("Analyze Statement", type="primary")
            
            if submitted:
                if not statement_text:
                    st.error("Please enter a statement to analyze.")
                    return
                
                with st.spinner("Analyzing statement..."):
                    make_prediction(statement_text)
                    
                if context:
                    st.markdown("### Statement Context")
                    st.info(context)
    
    with test_tab:
        st.markdown("""
        Analyze examples from our test dataset to evaluate svm model performance
        on real-world statements.
                    
        Support Vector Machine (SVM) finds the optimal boundary to separate true and false statements. It's particularly effective for text classification tasks and high-dimensional data like TF-IDF features.
        """)
        
        if st.button("Analyze Test Examples", type="primary"):
            with st.spinner("Analyzing test examples..."):
                analyze_random_test_data()

def make_prediction(statement_text):
    """Make predictions using SVM model"""
    try:
        # Preprocess the statement
        processed_text = preprocess_statement(statement_text)
        
        # Get model from session state
        model = st.session_state.model
        
        # Check if model is a pipeline or standalone classifier
        if hasattr(model, 'named_steps'):
            # Model is a pipeline - pass processed text directly
            prediction = model.predict([processed_text])[0]
            
            # SVM with LinearSVC doesn't have predict_proba, use decision_function
            if hasattr(model.named_steps['clf'], 'decision_function'):
                decision_score = model.decision_function([processed_text])[0]
                confidence = min(0.95, max(0.55, abs(decision_score) / 2))
            else:
                confidence = 0.75
        else:
            # Model is a standalone classifier - use TF-IDF vectorizer
            vectorizer = st.session_state.vectorizer
            features = vectorizer.transform([processed_text])
            prediction = model.predict(features)[0]
            
            if hasattr(model, 'decision_function'):
                decision_score = model.decision_function(features)[0]
                confidence = min(0.95, max(0.55, abs(decision_score) / 2))
            else:
                confidence = 0.75
        
        # Display the prediction
        prediction_label = "True" if prediction else "False"
        box_color = "#d4edda" if prediction else "#f8d7da"
        text_color = "#155724" if prediction else "#721c24"
        
        st.markdown("### Support Vector Machine (SVM) Prediction")
        st.markdown(
            f"""<div style='background-color: {box_color}; color: {text_color}; 
            padding: 1rem; border-radius: 0.5rem;'>
            <h4 style='margin-top: 0;'>Prediction: {prediction_label}</h4>
            <p>Confidence: {confidence:.2%}</p>
            </div>""",
            unsafe_allow_html=True
        )
        
        return {
            'prediction': bool(prediction),
            'confidence': confidence
        }
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error("Debug info:")
        st.write(f"Model type: {type(st.session_state.model)}")
        if hasattr(st.session_state.model, 'named_steps'):
            st.write(f"Pipeline steps: {list(st.session_state.model.named_steps.keys())}")
            clf = st.session_state.model.named_steps['clf']
            st.write(f"Classifier type: {type(clf)}")
        return None

def analyze_random_test_data():
    columns = [
        'filename', 'label', 'statement', 'subject', 'speaker', 
        'job_title', 'state', 'party', 'true_counts', 'false_counts',
        'half_true_counts', 'barely_true_counts', 'pants_fire_counts', 'context'
    ]

    df = pd.read_csv('test.tsv', sep='\t', header=None, names=columns)

    # Load test data
    test_df = clean_data(df)
    if test_df is None:
        st.error("Failed to load test data")
        return
    
    # Get 5 random samples
    test_samples = test_df.sample(n=min(5, len(test_df)))
    
    for _, row in test_samples.iterrows():
        st.markdown("---")
        st.markdown("### Test Statement to Analyze")
        st.markdown(f"> _{row['statement']}_")
        st.markdown(f"**True Label:** {row['label']}")
        st.markdown(f"**Context:** {row['context']}")
        
        make_prediction(row['statement'])

def main():
    st.markdown('<h1 class="main-header">Fake News Detection - SVM Classifier</h1>', unsafe_allow_html=True)
    
    initialize_app()

    tab = st.tabs(["SVM Analysis"])
    
    with tab[0]:
        detailed_analysis_tab()

if __name__ == "__main__":
    main()