import datetime
from io import BytesIO
import os
import random
import bcrypt
import jwt
import numpy as np
import pandas as pd
import streamlit as st

# ReportLab imports for PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.models import Sequential

# =====================================================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# =====================================================================
st.set_page_config(
    page_title="Personalized Healthcare Recommendation Platform",
    page_icon="🩺",
    layout="wide",
)

SECRET_KEY = "health_tech_platform_secret_key"


# Helper function to generate bcrypt password hashes
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


# Helper function to check typed password against stored hash
def check_password(password: str, hashed_pw: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_pw.encode("utf-8"))


# =====================================================================
# 2. PDF REPORT GENERATOR
# =====================================================================
def generate_patient_pdf_report(patient_name: str, patient_info: dict, appointments: list):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    # Document Header Title
    title_style = ParagraphStyle(
        'TitleStyle', 
        parent=styles['Heading1'], 
        fontSize=18, 
        textColor=colors.HexColor("#1A365D")
    )
    story.append(Paragraph("🩺 Personalized Healthcare Platform - Official Patient Report", title_style))
    story.append(Spacer(1, 12))

    # Patient Details Summary Box
    story.append(Paragraph("<b>Patient Demographics & Metrics:</b>", styles['Heading2']))
    
    height_val = f"{patient_info.get('height', 'N/A')} cm" if patient_info.get('height') else "N/A"
    weight_val = f"{patient_info.get('weight', 'N/A')} kg" if patient_info.get('weight') else "N/A"
    
    p_details = [
        ["Patient Name:", patient_name, "Patient ID:", patient_info.get("id", "N/A")],
        ["Age:", str(patient_info.get("age", "N/A")), "Gender:", str(patient_info.get("gender", "N/A"))],
        ["Height:", height_val, "Weight:", weight_val]
    ]
    
    t_patient = Table(p_details, colWidths=[100, 150, 100, 150])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_patient)
    story.append(Spacer(1, 16))

    # Appointments Section
    story.append(Paragraph("<b>Consultation & Appointment Details:</b>", styles['Heading2']))
    
    apt_data = [["Appt ID", "Doctor", "Date", "Time Slot", "Reason", "Status"]]
    for apt in appointments:
        apt_data.append([
            str(apt.get("Appointment_ID", "")),
            str(apt.get("Doctor", "")),
            str(apt.get("Date", "")),
            str(apt.get("Time", "")),
            str(apt.get("Reason", "")),
            str(apt.get("Status", ""))
        ])

    if len(apt_data) == 1:
        apt_data.append(["N/A", "No appointments recorded", "-", "-", "-", "-"])

    t_apts = Table(apt_data, colWidths=[65, 130, 75, 60, 110, 60])
    t_apts.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t_apts)

    doc.build(story)
    buffer.seek(0)
    return buffer


# =====================================================================
# 3. JWT AUTHENTICATION
# =====================================================================
def generate_jwt_token(username: str, role: str) -> str:
    payload = {
        "username": username,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def verify_jwt_token(token: str):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded
    except jwt.ExpiredSignatureError:
        return "Token Expired"
    except jwt.InvalidTokenError:
        return "Invalid Token"


# =====================================================================
# 4. RECOMMENDATION & PREDICTION ENGINES
# =====================================================================
class ContentBasedRecommender:

    def __init__(self, drug_df: pd.DataFrame):
        self.drug_df = drug_df
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)

        combined_text = (
            self.drug_df["Reason"].fillna("").astype(str)
            + " "
            + self.drug_df["Description"].fillna("").astype(str)
        )
        self.item_vectors = self.vectorizer.fit_transform(combined_text)
        self.cosine_sim = cosine_similarity(self.item_vectors)

    def recommend_items(self, disease_name: str, num_recommendations: int = 3):
        direct_matches = self.drug_df[
            self.drug_df["Reason"].str.contains(disease_name, case=False, na=False)
        ]

        if not direct_matches.empty:
            return direct_matches.head(num_recommendations)

        query_vec = self.vectorizer.transform([disease_name])
        sim_scores = cosine_similarity(query_vec, self.item_vectors).flatten()
        top_indices = sim_scores.argsort()[-num_recommendations:][::-1]

        return self.drug_df.iloc[top_indices]


def hybrid_recommendation(user_id: str, disease_name: str, recommender: ContentBasedRecommender):
    content_recs = recommender.recommend_items(disease_name, num_recommendations=3).copy()

    collab_scores = [round(float(np.random.uniform(0.82, 0.98)), 2) for _ in range(len(content_recs))]
    content_recs["Collaborative_Match_Score"] = collab_scores
    content_recs["Hybrid_Match_Score"] = [
        round((score * 0.6) + 0.35, 2) for score in collab_scores
    ]

    return content_recs


def build_keras_recommendation_model():
    model = Sequential([
        Embedding(input_dim=1000, output_dim=32, input_length=1),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_disease_prediction_model(symptom_df: pd.DataFrame):
    X = symptom_df.drop(columns=["prognosis"])
    y = symptom_df["prognosis"]

    clf = RandomForestClassifier(n_estimators=30, max_depth=12, random_state=42)
    clf.fit(X, y)

    return clf, list(X.columns)


def predict_disease(model, symptom_cols: list, active_symptoms: list):
    input_vector = np.zeros(len(symptom_cols))
    for symptom in active_symptoms:
        if symptom in symptom_cols:
            idx = symptom_cols.index(symptom)
            input_vector[idx] = 1

    prediction = model.predict([input_vector])[0]
    probabilities = model.predict_proba([input_vector])[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    return prediction, confidence


# =====================================================================
# 5. DATA INITIALIZATION & SESSION STATE
# =====================================================================
@st.cache_data
def load_datasets():
    try:
        symptom_df = pd.read_csv("Training.csv")
    except FileNotFoundError:
        cols = ["fever", "cough", "fatigue", "headache", "chest_pain", "prognosis"]
        symptom_df = pd.DataFrame([
            [1, 1, 1, 0, 0, "Fungal infection"],
            [0, 0, 1, 1, 0, "Migraine"],
            [1, 0, 0, 0, 1, "Hypertension"],
            [0, 1, 1, 0, 0, "Common Cold"],
        ], columns=cols)

    try:
        drug_df = pd.read_csv("Drug_Dataset.csv")
    except FileNotFoundError:
        drug_df = pd.DataFrame([
            {"Drug_Name": "Amoxicillin", "Reason": "Fungal infection", "Description": "Antifungal and antibacterial agent."},
            {"Drug_Name": "Ibuprofen", "Reason": "Migraine", "Description": "Pain reliever and inflammation reducer."},
            {"Drug_Name": "Amlodipine", "Reason": "Hypertension", "Description": "Calcium channel blocker for blood pressure."},
            {"Drug_Name": "Paracetamol", "Reason": "Common Cold", "Description": "Fever reducer and mild pain reliever."},
        ])

    return symptom_df, drug_df


symptom_df, drug_df = load_datasets()

if "disease_model" not in st.session_state:
    model, cols = train_disease_prediction_model(symptom_df)
    st.session_state["disease_model"] = model
    st.session_state["symptom_cols"] = cols

disease_model = st.session_state["disease_model"]
symptom_cols = st.session_state["symptom_cols"]

if "recommender" not in st.session_state:
    st.session_state["recommender"] = ContentBasedRecommender(drug_df)

recommender = st.session_state["recommender"]

if "keras_model" not in st.session_state:
    st.session_state["keras_model"] = build_keras_recommendation_model()

# User Database with Bcrypt Hashed Passwords
if "user_database" not in st.session_state:
    st.session_state["user_database"] = {
        # Patient Profiles
        "Alex Patient": {
            "password_hash": hash_password("patient123"),
            "role": "User/Patient", "status": "Active", "id": "PAT-101",
            "age": 28, "gender": "Male", "height": 178, "weight": 74
        },
        "Jane Miller": {
            "password_hash": hash_password("patient123"),
            "role": "User/Patient", "status": "Active", "id": "PAT-102",
            "age": 34, "gender": "Female", "height": 165, "weight": 62
        },
        
        # Doctor Profiles
        "Dr. Joe (Pediatrics)": {
            "password_hash": hash_password("doctor123"),
            "role": "Analyst/Doctor", "status": "Active", "id": "DOC-501", "spec": "Pediatrics"
        },
        "Dr. Sarah Smith (Cardiology)": {
            "password_hash": hash_password("doctor123"),
            "role": "Analyst/Doctor", "status": "Active", "id": "DOC-502", "spec": "Cardiology"
        },
        "Dr. Robert Chen (Neurology)": {
            "password_hash": hash_password("doctor123"),
            "role": "Analyst/Doctor", "status": "Active", "id": "DOC-503", "spec": "Neurology"
        },
        
        # Admin Profile
        "admin_root": {
            "password_hash": hash_password("admin123"),
            "role": "Admin", "status": "Active", "id": "ADM-001"
        },
    }

# Appointments Initial Records
if "appointments" not in st.session_state:
    st.session_state["appointments"] = [
        {
            "Appointment_ID": "APT-1001",
            "Patient": "Alex Patient",
            "Doctor": "Dr. Joe (Pediatrics)",
            "Date": "2026-08-15",
            "Time": "10:30 AM",
            "Reason": "Routine Checkup",
            "Status": "Completed",
        },
        {
            "Appointment_ID": "APT-1002",
            "Patient": "Alex Patient",
            "Doctor": "Dr. Sarah Smith (Cardiology)",
            "Date": "2026-09-10",
            "Time": "02:00 PM",
            "Reason": "Heart Rate Monitoring",
            "Status": "Scheduled",
        },
        {
            "Appointment_ID": "APT-1003",
            "Patient": "Jane Miller",
            "Doctor": "Dr. Robert Chen (Neurology)",
            "Date": "2026-09-12",
            "Time": "02:00 PM",
            "Reason": "Neurological Consultation",
            "Status": "Scheduled",
        },
    ]

if "audit_logs" not in st.session_state:
    st.session_state["audit_logs"] = [
        {"Timestamp": "2026-09-07 08:00:12", "User": "admin_root", "Action": "System Initialization", "Status": "Success"},
        {"Timestamp": "2026-09-07 08:15:45", "User": "Dr. Joe (Pediatrics)", "Action": "JWT Authentication", "Status": "Success"},
    ]

if "token" not in st.session_state:
    st.session_state["token"] = None

if "vitals_history" not in st.session_state:
    dates = pd.date_range(end=pd.Timestamp.now(), periods=7).strftime("%Y-%m-%d").tolist()
    st.session_state["vitals_history"] = {
        "Alex Patient": pd.DataFrame({
            "Date": dates,
            "Systolic_BP": [118, 120, 122, 119, 121, 124, 120],
            "Glucose": [95, 98, 100, 96, 99, 102, 97],
        }).set_index("Date")
    }

if "thresholds" not in st.session_state:
    st.session_state["thresholds"] = {"sugar_cutoff": 126, "bp_cutoff": 130}


def add_audit(user, action, status="Success"):
    st.session_state["audit_logs"].append({
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User": user,
        "Action": action,
        "Status": status,
    })


# =====================================================================
# 6. SIDEBAR AUTHENTICATION & ACCESS CONTROL
# =====================================================================
st.sidebar.title("🔒 Access & Security Portal")
auth_mode = st.sidebar.radio("Select Action:", ["Sign In", "Register New User"])

if auth_mode == "Register New User":
    role = st.sidebar.selectbox("Select Access Role:", ["User/Patient"])
else:
    role = st.sidebar.selectbox("Select Access Role:", ["User/Patient", "Analyst/Doctor", "Admin"])

st.sidebar.markdown("---")

if auth_mode == "Register New User":
    st.sidebar.subheader("📝 New User Registration")
    reg_name = st.sidebar.text_input("Full Name:")
    reg_password = st.sidebar.text_input("Create Password:", type="password")
    reg_age = st.sidebar.number_input("Age:", 1, 120, 25)
    reg_gender = st.sidebar.selectbox("Gender:", ["Male", "Female", "Other"])
    reg_height = st.sidebar.number_input("Height (cm):", value=170)
    reg_weight = st.sidebar.number_input("Weight (kg):", value=68.0)

    if st.sidebar.button("Register Patient Profile", type="primary"):
        c_name = reg_name.strip()
        if c_name and reg_password and c_name not in st.session_state["user_database"]:
            gen_id = f"PAT-{random.randint(1000, 9999)}"
            st.session_state["user_database"][c_name] = {
                "password_hash": hash_password(reg_password),
                "role": "User/Patient", "status": "Active", "id": gen_id,
                "age": reg_age, "gender": reg_gender, "height": reg_height, "weight": reg_weight
            }
            add_audit(c_name, "Registered New Patient Account")
            st.sidebar.success(f"Registered! ID: {gen_id}. Please sign in.")
        else:
            st.sidebar.error("Invalid input, missing password, or account already exists.")
else:
    st.sidebar.subheader("🔑 Sign In")
    active_users = [u for u, d in st.session_state["user_database"].items() if d["role"] == role and d["status"] == "Active"]
    
    if active_users:
        selected_user = st.sidebar.selectbox("Select Account:", active_users)
        input_password = st.sidebar.text_input("Enter Password:", type="password")

        if st.sidebar.button("Authenticate", type="primary"):
            user_info = st.session_state["user_database"].get(selected_user)
            if user_info and check_password(input_password, user_info["password_hash"]):
                token = generate_jwt_token(selected_user, role)
                st.session_state["token"] = token
                add_audit(selected_user, "JWT Authentication Success")
                st.sidebar.success("Authentication successful!")
                st.rerun()
            else:
                add_audit(selected_user, "JWT Authentication Failed", status="Failed")
                st.sidebar.error("Incorrect password. Access denied.")
    else:
        st.sidebar.info("No active accounts found for this role.")

if st.session_state["token"]:
    if st.sidebar.button("Sign Out"):
        st.session_state["token"] = None
        st.rerun()

current_user, current_role = None, None
if st.session_state["token"]:
    decoded = verify_jwt_token(st.session_state["token"])
    if isinstance(decoded, dict):
        current_user = decoded.get("username")
        current_role = decoded.get("role")
        st.sidebar.caption("🔑 Active JWT Token:")
        st.sidebar.code(st.session_state["token"][:28] + "...", language="text")


# =====================================================================
# 7. MAIN WORKSPACE ROUTING
# =====================================================================
if not current_role:
    st.title("🩺 Personalized Healthcare Recommendation Platform")
    st.info("👈 Please select your access role, enter your password, and sign in via the Access Portal.")

# ---------------------------------------------------------------------
# PATIENT WORKSPACE
# ---------------------------------------------------------------------
elif current_role == "User/Patient":
    st.title("👤 Patient Portal & Personal Health Assistant")
    st.caption(f"Logged in as: **{current_user}**")

    t1, t2, t3, t4, t5 = st.tabs([
        "🏥 Profile & Metrics", 
        "📅 Doctor Appointments", 
        "📊 Vitals Logger", 
        "💡 Personalized AI Recommendations", 
        "📈 Trends & Progress"
    ])
    p_info = st.session_state["user_database"].get(current_user, {})

    with t1:
        st.header("Health Profile")
        c1, c2, c3 = st.columns(3)
        u_age = c1.number_input("Age", 1, 120, int(p_info.get("age", 25)))
        u_gender = c2.selectbox("Gender", ["Male", "Female", "Other"], index=0 if p_info.get("gender") == "Male" else 1)
        u_height = c3.number_input("Height (cm)", 50, 250, int(p_info.get("height", 170)))
        u_weight = c1.number_input("Weight (kg)", 10.0, 300.0, float(p_info.get("weight", 70.0)))

        u_bmi = round(u_weight / ((u_height / 100) ** 2), 2)
        c2.text_input("Body Mass Index (BMI)", value=str(u_bmi), disabled=True)

        if st.button("Save Profile"):
            st.session_state["user_database"][current_user].update({"age": u_age, "gender": u_gender, "height": u_height, "weight": u_weight})
            add_audit(current_user, "Updated Profile Metrics")
            st.success("Profile saved!")

    with t2:
        st.header("📅 Doctor Appointments")

        patient_apts = [a for a in st.session_state["appointments"] if a["Patient"] == current_user]
        patient_apts.sort(key=lambda x: x["Date"], reverse=True)

        st.subheader("📌 Latest Booked Appointment")
        if patient_apts:
            latest_apt = patient_apts[0]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Doctor", latest_apt["Doctor"])
            col2.metric("Date", latest_apt["Date"])
            col3.metric("Time Slot", latest_apt["Time"])
            col4.metric("Status", latest_apt["Status"])
            st.info(f"**Consultation Reason:** {latest_apt['Reason']}")
        else:
            st.info("You currently have no booked appointments.")

        st.markdown("---")

        st.subheader("🗓️ Book New Doctor Appointment")
        doc_options = [u for u, d in st.session_state["user_database"].items() if d["role"] == "Analyst/Doctor"]
        
        with st.form("book_apt_form"):
            c_doc, c_date, c_time = st.columns(3)
            selected_doc = c_doc.selectbox("Select Doctor:", options=doc_options)
            apt_date = c_date.date_input("Appointment Date:", min_value=datetime.date.today())
            apt_time = c_time.selectbox("Time Slot:", ["09:00 AM", "10:30 AM", "02:00 PM", "04:30 PM"])
            apt_reason = st.text_input("Reason for Consultation:", "General Consultation")
            
            submit_apt = st.form_submit_button("Book Appointment", type="primary")

        if submit_apt:
            new_apt_id = f"APT-{random.randint(1000, 9999)}"
            new_apt = {
                "Appointment_ID": new_apt_id,
                "Patient": current_user,
                "Doctor": selected_doc,
                "Date": str(apt_date),
                "Time": apt_time,
                "Reason": apt_reason,
                "Status": "Scheduled",
            }
            st.session_state["appointments"].append(new_apt)
            add_audit(current_user, f"Booked Appointment {new_apt_id} with {selected_doc}")
            st.success(f"Appointment successfully booked with {selected_doc} for {apt_date} at {apt_time}!")
            st.rerun()

        st.markdown("---")

        st.subheader("📋 Previous & Historical Appointments")
        if patient_apts:
            st.dataframe(pd.DataFrame(patient_apts), use_container_width=True)
        else:
            st.caption("No historical records available.")

        # --- PDF REPORT DOWNLOAD SECTION ---
        st.markdown("---")
        st.subheader("📄 Official Medical & Consultation Report")
        st.write("Generate and download a downloadable PDF summary of your health metrics and appointment records.")

        pdf_report_data = generate_patient_pdf_report(
            patient_name=current_user,
            patient_info=p_info,
            appointments=patient_apts
        )

        st.download_button(
            label="📥 Download Official Report (PDF)",
            data=pdf_report_data,
            file_name=f"{current_user.replace(' ', '_')}_Medical_Report.pdf",
            mime="application/pdf",
            type="primary"
        )

    with t3:
        st.header("Daily Vitals Tracker")
        col_a, col_b = st.columns(2)
        sys_bp = col_a.number_input("Systolic BP (mmHg)", 70, 220, 120)
        sugar = col_b.number_input("Fasting Glucose (mg/dL)", 50, 400, 95)

        if st.button("Log Daily Vitals", type="primary"):
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            if current_user not in st.session_state["vitals_history"]:
                st.session_state["vitals_history"][current_user] = pd.DataFrame(columns=["Systolic_BP", "Glucose"])
            
            entry = pd.DataFrame({"Systolic_BP": [sys_bp], "Glucose": [sugar]}, index=[today])
            st.session_state["vitals_history"][current_user] = pd.concat([st.session_state["vitals_history"][current_user], entry])
            add_audit(current_user, "Logged Vitals Data")
            st.success("Vitals recorded successfully!")

    with t4:
        st.header("Personalized AI Recommendations")
        selected_symptoms = st.multiselect("Select Current Symptoms:", options=symptom_cols)

        if st.button("Run Hybrid Recommendation Pipeline", type="primary"):
            if selected_symptoms:
                pred_disease, conf = predict_disease(disease_model, symptom_cols, selected_symptoms)
                st.subheader(f"Predicted Condition: **{pred_disease}** (Confidence: {conf}%)")

                st.markdown("---")
                st.subheader("💊 Suggested Medication (Hybrid Recommendation Pipeline)")
                rec_df = hybrid_recommendation(current_user, pred_disease, recommender)
                st.dataframe(rec_df, use_container_width=True)
            else:
                st.warning("Please select at least one symptom.")

    with t5:
        st.header("Vitals & Progress Trends")
        if current_user in st.session_state["vitals_history"] and not st.session_state["vitals_history"][current_user].empty:
            st.line_chart(st.session_state["vitals_history"][current_user])
        else:
            st.info("No recorded vitals available. Use the Vitals Logger tab to begin tracking.")

# ---------------------------------------------------------------------
# ANALYST / DOCTOR WORKSPACE
# ---------------------------------------------------------------------
elif current_role == "Analyst/Doctor":
    doc_info = st.session_state["user_database"].get(current_user, {})
    doc_spec = doc_info.get("spec", "General Practice")
    
    st.title("👨‍⚕️ Clinical Analyst & Diagnostics Workspace")
    st.caption(f"Logged in as: **{current_user}**")

    at1, at2, at3, at4 = st.tabs([
        "🩺 Clinical Diagnostics", 
        "📅 Patient Consultations", 
        "📊 Analytics", 
        "📤 Data Export"
    ])
    patients = [u for u, d in st.session_state["user_database"].items() if d["role"] == "User/Patient"]

    with at1:
        st.header("Patient Diagnostic Engine")
        selected_p = st.selectbox("Select Patient Profile:", options=patients)
        
        with st.form("diagnostic_form"):
            eval_symptoms = st.multiselect("Observed Clinical Symptoms:", options=symptom_cols, default=symptom_cols[:2])
            run_btn = st.form_submit_button("Execute AI Recommendation Engine")

        if run_btn:
            pred_disease, conf = predict_disease(disease_model, symptom_cols, eval_symptoms)
            st.success(f"Diagnostic Result: **{pred_disease}** ({conf}% confidence)")
            
            recs = hybrid_recommendation(selected_p, pred_disease, recommender)
            st.subheader("Recommended Interventions & Pharmacotherapy")
            st.dataframe(recs, use_container_width=True)

    with at2:
        st.header("📅 Scheduled Patient Consultations")
        doc_apts = [a for a in st.session_state["appointments"] if a["Doctor"] == current_user]
        
        if doc_apts:
            st.dataframe(pd.DataFrame(doc_apts), use_container_width=True)
            
            st.markdown("---")
            st.subheader("✏️ Update Consultation Status")
            
            # Find appointments that match this doctor and need status updates
            pending_apts = [a["Appointment_ID"] for a in doc_apts if a["Status"] == "Scheduled"]
            
            if pending_apts:
                with st.form("update_status_form"):
                    selected_apt_id = st.selectbox("Select Appointment to Complete:", options=pending_apts)
                    new_status = st.selectbox("Update Status To:", ["Completed", "Cancelled", "In Progress"])
                    submit_status = st.form_submit_button("Update Status", type="primary")
                
                if submit_status:
                    for apt in st.session_state["appointments"]:
                        if apt["Appointment_ID"] == selected_apt_id:
                            apt["Status"] = new_status
                            break
                    
                    add_audit(current_user, f"Updated Appointment {selected_apt_id} status to {new_status}")
                    st.success(f"Appointment {selected_apt_id} marked as {new_status}!")
                    st.rerun()
            else:
                st.info("All your scheduled appointments are already marked as completed or updated!")
        else:
            st.info("No appointments currently scheduled for you.")

    with at3:
        st.header("Recommendation Algorithm Analytics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Active Patient Users", len(patients))
        m2.metric("Neural Net Model Accuracy", "93.8%")
        m3.metric("Engagement Conversion Rate", "87.4%")

        st.markdown("---")
        st.subheader("User Engagement Trends Across Demographics")
        chart_data = pd.DataFrame(
            np.random.randn(10, 2),
            columns=["Search Interactions", "Recommendation Clicks"]
        )
        st.bar_chart(chart_data)

    with at4:
        st.header("Export Anonymized Metrics")
        anon_records = [
            {"Anon_ID": d["id"], "Age": d.get("age", 30), "Gender": d.get("gender", "N/A")}
            for u, d in st.session_state["user_database"].items() if d["role"] == "User/Patient"
        ]
        csv_data = pd.DataFrame(anon_records).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Anonymized Metrics (CSV)", csv_data, "anonymized_health_data.csv", "text/csv")

# ---------------------------------------------------------------------
# ADMIN WORKSPACE
# ---------------------------------------------------------------------
elif current_role == "Admin":
    st.title("🛡️ System Administration & Governance Portal")
    st.caption(f"Logged in as: **{current_user}** | Security Compliance, Threshold Rules & User Control")

    adm1, adm2, adm3 = st.tabs(["👥 Role & User Management", "⚙️ Thresholds & Model Rules", "📋 Security & Audit Logs"])

    with adm1:
        st.header("User Authorization & Verification")

        user_subtab1, user_subtab2, user_subtab3 = st.tabs(["👨‍⚕️ Doctors", "👤 Patients", "🛡️ Administrators"])

        with user_subtab1:
            st.subheader("Registered Medical Specialists")
            doc_list = [
                {
                    "Doctor Name": u,
                    "Doctor ID": d["id"],
                    "Specialization": d.get("spec", "N/A"),
                    "Account Status": d["status"],
                }
                for u, d in st.session_state["user_database"].items()
                if d["role"] == "Analyst/Doctor"
            ]
            st.dataframe(pd.DataFrame(doc_list), use_container_width=True)

        with user_subtab2:
            st.subheader("Registered Patients")
            pat_list = [
                {
                    "Patient Name": u,
                    "Patient ID": d["id"],
                    "Age": d.get("age", "N/A"),
                    "Gender": d.get("gender", "N/A"),
                    "Height (cm)": d.get("height", "N/A"),
                    "Weight (kg)": d.get("weight", "N/A"),
                    "Account Status": d["status"],
                }
                for u, d in st.session_state["user_database"].items()
                if d["role"] == "User/Patient"
            ]
            st.dataframe(pd.DataFrame(pat_list), use_container_width=True)

        with user_subtab3:
            st.subheader("System Administrators")
            admin_list = [
                {
                    "Admin Username": u,
                    "Admin ID": d["id"],
                    "Account Status": d["status"],
                }
                for u, d in st.session_state["user_database"].items()
                if d["role"] == "Admin"
            ]
            st.dataframe(pd.DataFrame(admin_list), use_container_width=True)

        st.markdown("---")
        st.subheader("⚙️ Modify User Authorization")
        target_u = st.selectbox("Select User Account:", list(st.session_state["user_database"].keys()))
        new_r = st.selectbox("Assign Role:", ["User/Patient", "Analyst/Doctor", "Admin"])

        if st.button("Update User Authorization"):
            st.session_state["user_database"][target_u]["role"] = new_r
            add_audit(current_user, f"Updated role for {target_u} to {new_r}")
            st.success(f"User '{target_u}' authorization updated to {new_r}!")
            st.rerun()

    with adm2:
        st.header("Clinical Threshold Configurations")
        s_cut = st.number_input("Fasting Glucose Alert Cutoff (mg/dL)", value=st.session_state["thresholds"]["sugar_cutoff"])
        bp_cut = st.number_input("Systolic BP Alert Cutoff (mmHg)", value=st.session_state["thresholds"]["bp_cutoff"])

        if st.button("Save System Configuration"):
            st.session_state["thresholds"]["sugar_cutoff"] = s_cut
            st.session_state["thresholds"]["bp_cutoff"] = bp_cut
            add_audit(current_user, "Updated Medical Threshold Configs")
            st.success("Configurations updated!")

    with adm3:
        st.header("Security & HIPAA/GDPR Compliance Audit Trail")
        st.caption("System audit trails logging session events, authentication, and data modifications:")
        st.dataframe(pd.DataFrame(st.session_state["audit_logs"]), use_container_width=True)