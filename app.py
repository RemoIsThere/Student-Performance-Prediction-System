import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Student Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI improvement
st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #2e66ff;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1b4dff;
        box-shadow: 0 4px 12px rgba(27, 77, 255, 0.2);
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load('student_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

try:
    model, model_columns = load_model()
except Exception as e:
    st.error("Failed to load the prediction model. Please ensure 'student_model.pkl' and 'model_columns.pkl' exist.")
    st.stop()

# --- HEADER ---
st.title("🎓 Student Performance Risk Predictor")
st.markdown("### Optimized AI Model for Early Risk Detection")
st.markdown("---")

# --- SIDEBAR FOR INPUTS ---
st.sidebar.header("📋 Student Profile Inputs")
st.sidebar.markdown("Provide the student's background and behavioral data to predict their academic risk.")

def get_user_inputs():
    with st.sidebar:
        st.subheader("Demographics & Academic")
        age = st.slider("Age", 15, 22, 17, help="Student's current age")
        studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4], 
                                 format_func=lambda x: ["< 2 hours", "2 to 5 hours", "5 to 10 hours", "> 10 hours"][x-1],
                                 help="Time spent studying each week")
        failures = st.selectbox("Past Class Failures", [0, 1, 2, 3], help="Number of past class failures")
        absences = st.number_input("Absences", 0, 100, 0, help="Total school absences")
        
        st.subheader("Family Background")
        medu = st.selectbox("Mother's Education Level", [0, 1, 2, 3, 4], 
                            format_func=lambda x: ["None", "Primary", "Lower Secondary", "Upper Secondary", "Higher Education"][x],
                            index=2)
        fedu = st.selectbox("Father's Education Level", [0, 1, 2, 3, 4], 
                            format_func=lambda x: ["None", "Primary", "Lower Secondary", "Upper Secondary", "Higher Education"][x],
                            index=2)
        
        st.subheader("Lifestyle & Health")
        goout = st.slider("Socializing Frequency (1-5)", 1, 5, 3, help="How often the student goes out with friends (1 = very low, 5 = very high)")
        freetime = st.slider("Free Time (1-5)", 1, 5, 3, help="Amount of free time after school (1 = very low, 5 = very high)")
        health = st.slider("Health Status (1-5)", 1, 5, 5, help="Current health status (1 = very bad, 5 = very good)")
        
    return {
        'age': age, 'absences': absences, 'failures': failures,
        'studytime': studytime, 'goout': goout, 'Medu': medu,
        'Fedu': fedu, 'health': health, 'freetime': freetime
    }

input_data = get_user_inputs()

# --- MAIN CONTENT AREA ---
st.write("### 🔍 Review Input Profile")
st.info("Adjust the parameters in the left sidebar to simulate different student profiles and see how the prediction changes.")

# Display quick metrics
col_a, col_b, col_c = st.columns(3)
col_a.metric("Age", input_data['age'])
col_b.metric("Absences", input_data['absences'])
col_c.metric("Past Failures", input_data['failures'])

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Analyze Student Risk Profile"):
    with st.spinner('Analyzing patterns...'):
        input_df = pd.DataFrame([input_data])
        # Re-index to match the columns expected by the model
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        res_col1, res_col2 = st.columns([2, 1])
        
        if prediction == 1:
            with res_col1:
                st.error("#### ⚠️ Risk Status: **AT-RISK**")
                st.write("This student shows behavioral and academic patterns consistent with high risk of poor performance (e.g., scoring below passing thresholds). Early intervention is highly recommended.")
            with res_col2:
                st.metric(label="AI Confidence Score", value=f"{prob[1]*100:.1f}%", delta="-High Risk", delta_color="inverse")
        else:
            with res_col1:
                st.success("#### ✅ Risk Status: **ON-TRACK**")
                st.write("This student's profile suggests they are currently on track for academic success. Continue monitoring periodically.")
            with res_col2:
                st.metric(label="AI Confidence Score", value=f"{prob[0]*100:.1f}%", delta="Low Risk", delta_color="normal")
                
        # Feature importance / Recommendations
        st.markdown("<br>### 💡 Actionable Insights", unsafe_allow_html=True)
        recommendations = []
        if input_data['absences'] >= 10:
            recommendations.append("📉 **High Absences Detected**: Monitor attendance closely and engage with parents if possible.")
        if input_data['failures'] > 0:
            recommendations.append("📚 **Past Failures**: Consider specialized tutoring to address foundational knowledge gaps.")
        if input_data['studytime'] < 2:
            recommendations.append("⏱️ **Low Study Time**: Encourage participation in structured after-school study programs.")
        if input_data['goout'] >= 4 and input_data['studytime'] <= 2:
            recommendations.append("⚖️ **Work-Play Imbalance**: High socializing with low study time. Promote better time management strategies.")
            
        if recommendations:
            for rec in recommendations:
                st.warning(rec)
        else:
            st.info("👍 No immediate red flags detected based on individual behavioral metrics. Maintain current habits.")
