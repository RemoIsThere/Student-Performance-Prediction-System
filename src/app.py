import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import tempfile
from fpdf import FPDF
import base64
import os

# Base paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "student_model.pkl")
COLUMNS_PATH = os.path.join(ROOT_DIR, "models", "model_columns.pkl")
DATA_PATH = os.path.join(ROOT_DIR, "data", "student-mat.csv")

# Set Page Config
st.set_page_config(
    page_title="Student Risk Forecaster",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minor tweaks (majority of theming is in config.toml)
st.markdown("""
    <style>
    .stDownloadButton > button {
        width: 100%;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    /* Hide index in pandas dataframe rendering */
    .row_heading.level0 {display:none}
    .blank {display:none}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    # Let's also load the dataset to compute baselines for radar charts
    if os.path.exists(DATA_PATH):
        df_raw = pd.read_csv(DATA_PATH, sep=';')
    else:
        df_raw = None
    return model, columns, df_raw

try:
    model, model_columns, df_raw = load_model()
except Exception as e:
    st.error("Failed to load the prediction model. Please ensure 'student_model.pkl' and 'model_columns.pkl' exist.")
    st.stop()

# Helper function to create PDF
def create_pdf_report(student_data, prediction_status, confidence_str, recommendations):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Student Intervention Report", ln=True, align='C')
    pdf.ln(10)
    
    # Status
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, txt=f"Risk Status: {prediction_status}", ln=False, align='L')
    pdf.cell(100, 10, txt=f"Confidence: {confidence_str}", ln=True, align='R')
    pdf.ln(5)
    
    # Student Profile
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Student Profile Data", ln=True, align='L')
    pdf.set_font("Arial", '', 10)
    
    for key, value in student_data.items():
        pdf.cell(100, 8, txt=f"{key.capitalize()}: {value}", ln=True, align='L')
        
    pdf.ln(10)
    
    # Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Actionable Insights & Recommendations", ln=True, align='L')
    pdf.set_font("Arial", '', 10)
    
    if recommendations:
        for rec in recommendations:
            # removing emojis for PDF to avoid character encoding issues
            clean_rec = rec.encode('ascii', 'ignore').decode('ascii')
            pdf.multi_cell(0, 10, txt=f"- {clean_rec}")
    else:
        pdf.cell(200, 10, txt="- No immediate red flags detected. Maintain current habits.", ln=True)

    # Save to temp file
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    pdf.output(path)
    return path

# Helper function to generate radar chart
def plot_radar_chart(student_profile, baseline_means):
    # Normalize features to 0-1 scale for radar chart display
    features = ['absences', 'failures', 'studytime', 'goout', 'health']
    
    # Max values for normalization (approx based on dataset)
    max_vals = {'absences': 30, 'failures': 4, 'studytime': 4, 'goout': 5, 'health': 5}
    
    student_norm = [min(student_profile.get(f, 0) / max_vals[f], 1.0) for f in features]
    
    if baseline_means is not None:
        baseline_norm = [min(baseline_means.get(f, 0) / max_vals[f], 1.0) for f in features]
    else:
        baseline_norm = [0.2, 0.1, 0.5, 0.6, 0.8] # Mock fallbacks
        
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=baseline_norm,
        theta=features,
        fill='toself',
        name='Cohort Average',
        line_color='rgba(255, 255, 255, 0.4)',
        fillcolor='rgba(255, 255, 255, 0.1)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=student_norm,
        theta=features,
        fill='toself',
        name='Current Student',
        line_color='#00B4D8',
        fillcolor='rgba(0, 180, 216, 0.4)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        margin=dict(l=40, r=40, t=20, b=20),
        font=dict(color='white')
    )
    return fig

# --- HEADER ---
st.title("🎓 Intelligent Student Forecaster")
st.markdown("Advanced analytics platform for early student risk detection and intervention tracking.")

tab1, tab2 = st.tabs(["🔍 Individual Analysis", "📂 Batch Processing (Classroom)"])

with tab1:
    st.markdown("### Profile Constructor")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.subheader("Academics")
        age = st.number_input("Age", 15, 22, 17)
        studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4], 
                                 format_func=lambda x: {"1": "< 2h", "2": "2-5h", "3": "5-10h", "4": "> 10h"}[str(x)])
        failures = st.selectbox("Past Failures", [0, 1, 2, 3])
        absences = st.number_input("Total Absences", 0, 100, 0)
        
    with col_input2:
        st.subheader("Family Background")
        medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4], index=2, 
                            format_func=lambda x: ["None", "Primary", "Lower Sec", "Upper Sec", "Higher Ed"][x])
        fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4], index=2,
                            format_func=lambda x: ["None", "Primary", "Lower Sec", "Upper Sec", "Higher Ed"][x])
                            
    with col_input3:
        st.subheader("Health & Lifestyle")
        goout = st.slider("Socializing (1-5)", 1, 5, 3)
        freetime = st.slider("Free Time (1-5)", 1, 5, 3)
        health = st.slider("Health Status (1-5)", 1, 5, 5)

    input_data = {
        'age': age, 'absences': absences, 'failures': failures,
        'studytime': studytime, 'goout': goout, 'Medu': medu,
        'Fedu': fedu, 'health': health, 'freetime': freetime
    }

    if st.button("🚀 Execute Risk Analysis", type="primary"):
        with st.spinner('Running AI Inference...'):
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=model_columns, fill_value=0)
            
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            
            st.divider()
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.subheader("Status & Heuristics")
                
                if prediction == 1:
                    status_text = "AT-RISK"
                    st.error(f"### ⚠️ {status_text}")
                    conf_str = f"{prob[1]*100:.1f}%"
                    st.metric("Risk Probability", conf_str, delta="High Intervention Priority", delta_color="inverse")
                else:
                    status_text = "ON-TRACK"
                    st.success(f"### ✅ {status_text}")
                    conf_str = f"{prob[0]*100:.1f}%"
                    st.metric("Success Probability", conf_str, delta="Low Risk", delta_color="normal")

                st.markdown("#### Actionable Insights")
                recommendations = []
                if input_data['absences'] >= 10:
                    recommendations.append("📉 High Absences Detect: Setup parent-teacher consult.")
                if input_data['failures'] > 0:
                    recommendations.append("📚 Past Failures: Assign foundational tutoring block.")
                if input_data['studytime'] < 2:
                    recommendations.append("⏱️ Study Time Deficit: Assign to after-school study hall.")
                if input_data['goout'] >= 4 and input_data['studytime'] <= 2:
                    recommendations.append("⚖️ High Distraction: Counsel on time-management and priority setting.")
                    
                if recommendations:
                    for rec in recommendations:
                        st.warning(rec)
                else:
                    st.info("No immediate interventions required.")
                    
                # PDF Generation
                pdf_path = create_pdf_report(input_data, status_text, conf_str, recommendations)
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📄 Download Official PDF Report",
                        data=pdf_file,
                        file_name="Student_Intervention_Report.pdf",
                        mime="application/pdf"
                    )

            with res_col2:
                st.subheader("Comparative Analytics")
                baseline = df_raw.mean(numeric_only=True).to_dict() if df_raw is not None else None
                fig = plot_radar_chart(input_data, baseline)
                st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.markdown("### Classroom Batch Upload")
    st.markdown("Upload a CSV file containing student records. The file must contain columns corresponding to the profile metrics: `age, absences, failures, studytime, goout, Medu, Fedu, health, freetime`.")
    
    # We provide a sample template for them
    template_data = pd.DataFrame([{col: 0 for col in model_columns}])
    st.download_button("Download CSV Template", data=template_data.to_csv(index=False), file_name="student_template.csv", mime="text/csv")
    
    uploaded_file = st.file_uploader("Upload Classroom Data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        st.write(f"Loaded {len(batch_df)} student records.")
        # Reindex just in case
        valid_df = batch_df.reindex(columns=model_columns, fill_value=0)
        
        with st.spinner("Processing batch..."):
            predictions = model.predict(valid_df)
            probs = model.predict_proba(valid_df)
            
            batch_df['Risk_Status'] = ['At-Risk' if p == 1 else 'On-Track' for p in predictions]
            batch_df['Risk_Probability'] = [f"{prob[1]*100:.1f}%" for prob in probs]
            
            st.divider()
            
            col_b1, col_b2 = st.columns([1, 2])
            
            at_risk_count = sum(predictions == 1)
            safe_count = len(predictions) - at_risk_count
            
            with col_b1:
                st.metric("Total Flagged At-Risk", f"{at_risk_count} / {len(predictions)}")
                
                # Plotly Pie Chart for summary
                fig_pie = go.Figure(data=[go.Pie(labels=['On-Track', 'At-Risk'], 
                                                 values=[safe_count, at_risk_count],
                                                 hole=0.4,
                                                 marker_colors=['#10B981', '#EF4444'])])
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_b2:
                st.subheader("Intervention Roster")
                def highlight_risk(val):
                    color = 'red' if val == 'At-Risk' else 'green'
                    return f'color: {color}'
                
                st.dataframe(batch_df.style.applymap(highlight_risk, subset=['Risk_Status']), use_container_width=True, height=350)
                
                # Export results
                st.download_button(
                    "💾 Export Analyzed Roster",
                    data=batch_df.to_csv(index=False),
                    file_name="processed_classroom_roster.csv",
                    mime="text/csv"
                )
