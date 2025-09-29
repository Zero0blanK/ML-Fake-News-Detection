# Fake News Detection 🔍

An interactive machine learning application that analyzes political statements and predicts their truthfulness using multiple ML models.

## 📋 Overview

This project uses machine learning to analyze political statements and predict their truthfulness based on various features like speaker information, historical fact-checking data, and statement context. It implements multiple models including Linear Regression, Naive Bayes, and Random Forest classifiers.

## 🌟 Features

- **Multiple ML Models**:
  - Support Vector Machines
  - Naive Bayes
  - Random Forest Classifier

- **Interactive Analysis**:
  - Analysis: Instant prediction with minimal input
  - Detailed Analysis: Full control over all features
  - Data Overview: Dataset statistics and visualizations

## 🛠️ Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models**:
   - Run the Jupyter notebook:
     ```bash
     jupyter notebook train_models.ipynb
     ```
   - Execute all cells to train and save models

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## 📊 Data

The project uses the LIAR dataset which includes:
- Political statements
- Speaker information
- Historical fact-checking counts
- Context and metadata

LINK: https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset?select=README

### Data Files:
- `train.tsv`: Used for training the models
- `test.tsv`: Used for model evaluation

## 🎯 Usage

1. **Quick Analysis**:
   - Enter a political statement
   - Get instant predictions from 3 models (SVM, NB, RF)

2. **Detailed Analysis**:
   - Input statement text
   - Configure speaker information
   - Adjust historical fact-check counts
   - View comprehensive prediction results

3. **Data Overview**:
   - Explore dataset statistics
   - View data distributions
   - Analyze model performance

## 📈 Models

### Support Vector Machines
- Finds the best boundary to separate classes
- Effective in high-dimensional spaces
- Uses kernels to handle non-linear data

### Naive Bayes
- Fast and efficient
- Probability-based predictions
- Good for text classification

### Random Forest
- Ensemble learning method
- Feature importance analysis
- Robust predictions

## 🔄 Model Training

The training process (`train_models.ipynb`) includes:
1. Data preprocessing
2. Feature engineering and data splitting
3. Model training and validation
4. Performance evaluation
5. Save Model

## 🎨 UI Features

- Clean, modern interface
- Interactive visualizations
- Color-coded predictions
- Detailed probability distributions
- Responsive layout

## 📝 Project Structure

```
LIAR-ML-Project/
├── app.py                 # Streamlit application
├── train_models.ipynb     # Model training notebook
├── requirements.txt       # Project dependencies
├── models/               # Saved model files
│   ├── svm.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── encoders.joblib
│   └── scaler.joblib
├── train.tsv             # Training data
└── test.tsv              # Test data

