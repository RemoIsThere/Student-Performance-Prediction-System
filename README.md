# 🎓 Student Performance Prediction System

An AI-powered web application designed to predict whether a student is "At-Risk" of struggling academically, based on their demographic, lifestyle, and academic history data.

This project aims to help educators and institutions identify students who need early interventions, enabling targeted support to improve academic outcomes.

## 🌟 Key Features

- **Predictive Machine Learning Model**: Uses an optimized Random Forest Classifier to process student data and determine the risk level.
- **Interactive Web Interface**: A sleek, user-friendly Streamlit application where users can simulate student profiles.
- **Instant Insights**: Provides probabilistic confidence scores and direct recommendations based on identified risk factors (e.g., high absences, past failures).
- **Robust Training Pipeline**: Contains clean, modular code for training, evaluating, and exporting the ML model.

## 📊 Dataset

The dataset tracks student achievements in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social, and school-related features. Students with a final grade (`G3`) below the passing mark of 10 are classified as **At-Risk**.

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Machine Learning**: Scikit-Learn (Random Forest)
- **Data Manipulation**: Pandas
- **Web Framework**: Streamlit
- **Model Serialization**: Joblib

## 🚀 Quick Start & Installation

### 1. Clone the repository

```bash
git clone https://github.com/RemoIsThere/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Install dependencies

It's recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

The application will be accessible locally at `http://localhost:8501`.

## 🧠 Model Training

If you want to re-train the model with new data or hyperparameter adjustments:

```bash
python main.py
```

This script will preprocess the `student-mat.csv` data, retrain the Random Forest model, output accuracy metrics, and export the `.pkl` artifacts.

## 📂 Project Structure

```text
.
├── app.py                  # Streamlit web application
├── main.py                 # Model training pipeline and evaluation script
├── predict.py              # Command-line utility for predictions
├── requirements.txt        # Python dependencies
├── model_columns.pkl       # Serialized columns expected by the model
├── student_model.pkl       # Trained Scikit-Learn model
└── student-mat.csv         # Raw dataset
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a PR!

## 📝 License

This project is [MIT](https://opensource.org/licenses/MIT) licensed.
