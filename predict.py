"""
CLI utility for making predictions using the trained Student Performance Model.
"""

import pandas as pd
import joblib

class StudentRiskPredictor:
    def __init__(self, model_path='student_model.pkl', columns_path='model_columns.pkl'):
        try:
            self.model = joblib.load(model_path)
            self.model_columns = joblib.load(columns_path)
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            raise

    def predict(self, student_data):
        """
        Predict if a student is at risk based on their profile data.
        
        Args:
            student_data (dict): Dictionary mapping feature names to values.
            
        Returns:
            dict: Prediction status and confidence probabilities.
        """
        df = pd.DataFrame([student_data]).reindex(columns=self.model_columns, fill_value=0)
        pred = self.model.predict(df)[0]
        prob = self.model.predict_proba(df)[0]
        
        status = "AT-RISK" if pred == 1 else "ON-TRACK"
        confidence = prob[pred] * 100
        
        return {
            "status": status,
            "confidence": confidence,
            "probabilities": prob
        }

if __name__ == "__main__":
    predictor = StudentRiskPredictor()
    
    # Test profile with high absences and past failures
    sample_profile = {
        'absences': 18, 
        'failures': 2, 
        'age': 18, 
        'studytime': 1, 
        'goout': 4, 
        'Medu': 1, 
        'Fedu': 1, 
        'health': 3, 
        'freetime': 4
    }
    
    print("Evaluating Sample Student Profile...")
    result = predictor.predict(sample_profile)
    print(f"Result: {result['status']} | Confidence: {result['confidence']:.2f}%")
