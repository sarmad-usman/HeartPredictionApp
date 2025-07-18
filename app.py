import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.utils import ImageReader
from sklearn.metrics import accuracy_score
from datetime import datetime

# Load models, scaler, test data for accuracy calculation
scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("random_forest_model.pkl")
log_model = joblib.load("logistic_regression_model.pkl")

X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

# Calculate model accuracies
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
log_acc = accuracy_score(y_test, log_model.predict(X_test))

# Streamlit Page Config
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: crimson;'>❤️ Heart Disease Prediction</h1>
        <hr>
    </div>
""", unsafe_allow_html=True)

# Patient Name (Optional)
patient_name = st.text_input("👤 Patient Name (Optional)", "")

# Input Form
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ['Male', 'Female'])
    dataset = st.selectbox("Dataset Origin", ['Cleveland', 'Hungarian', 'Switzerland', 'VA'])
    cp = st.selectbox("Chest Pain Type", ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=250, value=120)
    chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['TRUE', 'FALSE'])

with col2:
    restecg = st.selectbox("Resting ECG", ['normal', 'st-t abnormality', 'lv hypertrophy'])
    thalch = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", ['TRUE', 'FALSE'])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", ['upsloping', 'flat', 'downsloping'])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ['normal', 'fixed defect', 'reversable defect'])

# Data preparation
def preprocess_input(df):
    df_copy = df.copy()
    mappings = {
        'sex': {'Male': 1, 'Female': 0},
        'cp': {'typical angina': 3, 'atypical angina': 2, 'non-anginal': 1, 'asymptomatic': 0},
        'fbs': {'TRUE': 1, 'FALSE': 0},
        'restecg': {'normal': 1, 'st-t abnormality': 2, 'lv hypertrophy': 0},
        'exang': {'TRUE': 1, 'FALSE': 0},
        'slope': {'upsloping': 2, 'flat': 1, 'downsloping': 0},
        'thal': {'normal': 2, 'fixed defect': 1, 'reversable defect': 0},
        'dataset': {'Cleveland': 0, 'Hungarian': 1, 'Switzerland': 2, 'VA': 3}
    }
    for col, mapping in mappings.items():
        df_copy[col] = df_copy[col].map(mapping)
    return df_copy

# Create dataframe for input
input_df = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'dataset': [dataset],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalch': [thalch],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

processed_input = preprocess_input(input_df)
scaled_input = scaler.transform(processed_input)

# Model selection and prediction
model_option = st.radio("Choose Prediction Model", ("Random Forest", "Logistic Regression"))

if model_option == "Random Forest":
    selected_model_accuracy = rf_acc
    selected_model_name = "Random Forest"
else:
    selected_model_accuracy = log_acc
    selected_model_name = "Logistic Regression"

st.markdown(f"📊 **Model Accuracy:** {selected_model_accuracy:.2%} | Model: {selected_model_name}")

if st.button("🔍 Predict"):
    if model_option == "Random Forest":
        result = rf_model.predict(scaled_input)[0]
    else:
        result = log_model.predict(scaled_input)[0]

    # Summary HTML
    summary_html = f"""
    <div style='background-color: #2a2d34; padding: 15px; border-radius: 10px; color: white; margin-top: 20px; margin-bottom:20px;'>
        <h4>🩺 Summary of Patient Data:</h4>
        <table style='width:100%; font-size:15px;'>
            <tr><td><b>Name:</b></td><td>{patient_name or 'N/A'}</td></tr>
            <tr><td><b>Age:</b></td><td>{age} years</td></tr>
            <tr><td><b>Sex:</b></td><td>{sex}</td></tr>
            <tr><td><b>Dataset:</b></td><td>{dataset}</td></tr>
            <tr><td><b>Chest Pain:</b></td><td>{cp}</td></tr>
            <tr><td><b>Resting BP:</b></td><td>{trestbps} mm Hg</td></tr>
            <tr><td><b>Cholesterol:</b></td><td>{chol} mg/dl</td></tr>
            <tr><td><b>Fasting Blood Sugar >120:</b></td><td>{fbs}</td></tr>
            <tr><td><b>Resting ECG:</b></td><td>{restecg}</td></tr>
            <tr><td><b>Max Heart Rate:</b></td><td>{thalch} bpm</td></tr>
            <tr><td><b>Exercise Induced Angina:</b></td><td>{exang}</td></tr>
            <tr><td><b>ST Depression:</b></td><td>{oldpeak}</td></tr>
            <tr><td><b>Slope of ST:</b></td><td>{slope}</td></tr>
            <tr><td><b>Major Vessels:</b></td><td>{ca}</td></tr>
            <tr><td><b>Thalassemia:</b></td><td>{thal}</td></tr>
        </table>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

    # Prediction message
    if result == 1:
        st.error("⚠️ Patient is likely to have Heart Disease.")
    else:
        st.success("✅ Patient is unlikely to have Heart Disease.")

    # Advice generation
    advice = []
    if fbs == 'TRUE':
        advice.append("Monitor your blood sugar levels and consult a diabetologist.")
    if exang == 'TRUE':
        advice.append("Avoid strenuous physical activity that can trigger angina.")
    if cp in ['asymptomatic', 'non-anginal']:
        advice.append("Get a detailed cardiac evaluation for unexplained chest symptoms.")
    if ca >= 2:
        advice.append("Multiple blocked vessels detected. Seek a cardiologist immediately.")
    if thal != 'normal':
        advice.append("Thalassemia signs present. Hematology consultation is advised.")
    if chol >= 240:
        advice.append("Your cholesterol level is high. Dietary changes and medication may be needed.")
    if not advice:
        advice.append("Maintain a healthy lifestyle. Regular checkups and physical activity are recommended.")

    st.markdown("""
        <div>
            <h4>📋 Personalized Advice:</h4>
            <ul>
    """, unsafe_allow_html=True)
    for point in advice:
        st.markdown(f"<li>{point}</li>", unsafe_allow_html=True)
    st.markdown("</ul></div>", unsafe_allow_html=True)

    # PDF generation function
    def generate_pdf(model_name, model_acc):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Optional watermark
        try:
            watermark = ImageReader("watermark.png")
            c.drawImage(watermark, 240, 260, width=400, preserveAspectRatio=True, mask='auto')
        except:
            pass

        # Header
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width / 2, height - 50, "Heart Disease Prediction Report")

        # Table data
        data = [
            ["Patient Data", ""],
            ["", ""],
            ["Name", patient_name or 'N/A'],
            ["Age", f"{age} years"],
            ["Sex", sex],
            ["Dataset", dataset],
            ["Chest Pain", cp],
            ["Resting BP", f"{trestbps} mm Hg"],
            ["Cholesterol", f"{chol} mg/dl"],
            ["Fasting Blood Sugar >120", fbs],
            ["Resting ECG", restecg],
            ["Max Heart Rate", f"{thalch} bpm"],
            ["Exercise Induced Angina", exang],
            ["ST Depression", str(oldpeak)],
            ["Slope of ST", slope],
            ["Major Vessels", str(ca)],
            ["Thalassemia", thal]
        ]

        table = Table(data, colWidths=[200, 300])
        style = TableStyle([
            ('SPAN', (0, 0), (1, 0)),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('FONTNAME', (0, 2), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 2), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 2), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('ALIGN', (0, 2), (-1, -1), 'LEFT')
        ])
        table.setStyle(style)

        table_top = height - 130
        table.wrapOn(c, width, table_top)
        table.drawOn(c, 50, table_top - (len(data) * 18))

        # Prediction line below table
        prediction_y = table_top - (len(data) * 18) - 30
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.red if result == 1 else colors.green)
        prediction_text = f"Prediction: {'Likely to have Heart Disease' if result == 1 else 'Unlikely to have Heart Disease'}"
        c.drawString(50, prediction_y, prediction_text)

        # Advice section below prediction
        advice_y = prediction_y - 30
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.black)
        c.drawString(50, advice_y, "Personalized Advice:")
        advice_y -= 18
        c.setFont("Helvetica", 11)
        for point in advice:
            c.drawString(60, advice_y, f"- {point}")
            advice_y -= 16

        # Model info - bottom left
        model_info_y = 50
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.HexColor("#FF8C00"))  # Dark orange
        model_info = f"Model: {model_name} | Accuracy: {model_acc:.2%}"
        c.drawString(50, model_info_y, model_info)

        # Report Generated On - bottom right
        datetime_y = 50
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.HexColor("#555555"))  # Dark gray
        report_datetime = datetime.now().strftime("Report Generated On: %Y-%m-%d %H:%M:%S")
        text_width = c.stringWidth(report_datetime, "Helvetica", 10)
        c.drawString(width - text_width - 50, datetime_y, report_datetime)

        # Footer centered
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColor(colors.darkred)
        c.drawCentredString(width / 2, 20, "This report is system-generated by Sarmad Medical Intelligence Suite.")

        c.save()
        buffer.seek(0)
        return buffer

    # Download PDF button
    pdf_bytes = generate_pdf(selected_model_name, selected_model_accuracy)
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"HeartDiseaseReport_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf",
        mime='application/pdf'
    )

# --------- Footer ---------
st.markdown(
    """
    <style>
    .footer {
        width: 100%;
        background-color: #0E1117;  /* Match Streamlit dark background */
        color: #888888;
        text-align: center;
        padding: 10px 0;
        font-size: 13px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-top: 1px solid #2E3038;  /* subtle border */
        margin-top: 40px;
    }
    </style>
    <div class="footer">
        © 2025 Sarmad Medical Intelligence Suite — by M Sarmad Usman
    </div>
    """,
    unsafe_allow_html=True,
)
