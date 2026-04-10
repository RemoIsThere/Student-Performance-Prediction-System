"""
Student Performance Prediction Model Training Pipeline

This script loads the student performance dataset, preprocesses features,
trains a Random Forest classifier, evaluates its accuracy, and saves
the model artifacts for deployment.
"""

import os
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configure paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'student-mat.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'student_model.pkl')
COLUMNS_PATH = os.path.join(ROOT_DIR, 'models', 'model_columns.pkl')

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    logging.info("Starting model training pipeline...")

    # --- STEP 1: DATA LOADING ---
    try:
        df = pd.read_csv(DATA_PATH, sep=';')
        logging.info(f"Successfully loaded dataset with {df.shape[0]} records.")
    except FileNotFoundError:
        logging.error(f"Dataset '{DATA_PATH}' not found. Please ensure it exists.")
        return

    # --- STEP 2: PREPROCESSING ---
    # Define 'At-Risk' as G3 (final grade) < 10 (pass mark in many European systems out of 20)
    df['at_risk'] = (df['G3'] < 10).astype(int)

    # FEATURE SELECTION: Focusing on actionable and significant features
    selected_features = [
        'absences', 'failures', 'age', 'studytime', 
        'goout', 'Medu', 'Fedu', 'health', 'freetime'
    ]
    X = df[selected_features]
    y = df['at_risk']

    # --- STEP 3: TRAINING ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets.")

    # Hyperparameter Tuning
    model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=12, 
        min_samples_split=5, 
        random_state=42,
        class_weight='balanced'
    )
    
    logging.info("Training Random Forest Classifier...")
    model.fit(X_train, y_train)

    # --- STEP 4: EVALUATION ---
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logging.info(f"Model Accuracy Validation: {accuracy:.2%}")
    print("\nClassification Report:\n", classification_report(y_test, predictions))

    # --- STEP 5: SAVE ARTIFACTS ---
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(selected_features, COLUMNS_PATH)
        logging.info("Optimized model and feature columns saved successfully to disk.")
    except Exception as e:
        logging.error(f"Failed to save model artifacts: {e}")

if __name__ == "__main__":
    train_model()
