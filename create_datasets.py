import numpy as np
import pandas as pd

print("🚀 Generating Healthcare & Drug Datasets...")

# =====================================================================
# 1. CREATE DISEASE & SYMPTOM DATASET (Training.csv)
# =====================================================================
symptoms = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
    "continuous_sneezing",
    "shivering",
    "chills",
    "joint_pain",
    "stomach_pain",
    "acidity",
    "ulcers_on_tongue",
    "muscle_wasting",
    "vomiting",
    "burning_micturition",
    "spotting_urination",
    "fatigue",
    "weight_gain",
    "anxiety",
    "cold_hands_and_feets",
    "mood_swings",
    "weight_loss",
    "restlessness",
    "lethargy",
    "patches_in_throat",
    "irregular_sugar_level",
    "cough",
    "high_fever",
    "sunken_eyes",
    "breathlessness",
    "sweating",
    "dehydration",
    "indigestion",
    "headache",
    "yellowish_skin",
    "dark_urine",
    "nausea",
    "loss_of_appetite",
    "pain_behind_the_eyes",
    "back_pain",
    "constipation",
    "abdominal_pain",
    "diarrhoea",
    "mild_fever",
    "yellowing_of_eyes",
    "swelled_lymph_nodes",
    "malaise",
    "blurred_and_distorted_vision",
    "phlegm",
    "throat_irritation",
    "redness_of_eyes",
    "sinus_pressure",
    "runny_nose",
    "congestion",
    "chest_pain",
    "weakness_in_limbs",
    "fast_heart_rate",
    "pain_during_bowel_movements",
    "pain_in_anal_region",
    "bloody_stool",
    "irritation_in_anus",
    "neck_pain",
    "dizziness",
    "cramps",
    "bruising",
    "obesity",
    "swollen_legs",
    "swollen_blood_vessels",
    "puffy_face_and_eyes",
    "enlarged_thyroid",
    "brittle_nails",
    "swollen_extremeties",
    "excessive_hunger",
    "extra_marital_contacts",
    "drying_and_tingling_lips",
    "slurred_speech",
    "knee_pain",
    "hip_joint_pain",
    "muscle_weakness",
    "stiff_neck",
    "swelling_joints",
    "movement_stiffness",
    "spinning_movements",
    "loss_of_balance",
    "unsteadiness",
    "weakness_of_one_body_side",
    "loss_of_smell",
    "bladder_discomfort",
    "foul_smell_of_urine",
    "continuous_feel_of_urine",
    "passage_of_gases",
    "internal_itching",
    "toxic_look_(typhos)",
    "depression",
    "irritability",
    "muscle_pain",
    "altered_sensorium",
    "red_spots_over_body",
    "belly_pain",
    "abnormal_menstruation",
    "dischromic_patches",
    "watering_from_eyes",
    "increased_appetite",
    "polyuria",
    "family_history",
    "mucoid_sputum",
    "rusty_sputum",
    "lack_of_concentration",
    "visual_disturbances",
    "receiving_blood_transfusion",
    "receiving_unsterile_injections",
    "coma",
    "stomach_bleeding",
    "distention_of_abdomen",
    "history_of_alcohol_consumption",
    "fluid_overload",
    "blood_in_sputum",
    "prominent_veins_on_calf",
    "palpitations",
    "painful_walking",
    "pus_filled_blisters",
    "blackheads",
    "scurring",
    "skin_peeling",
    "silver_like_dusting",
    "small_dents_in_nails",
    "inflammatory_nails",
    "blister",
    "red_sore_around_nose",
    "yellow_crust_ooze",
]

diseases = [
    "Fungal infection",
    "Allergy",
    "GERD",
    "Chronic cholestasis",
    "Drug Reaction",
    "Peptic ulcer diseae",
    "AIDS",
    "Diabetes",
    "Gastroenteritis",
    "Bronchial Asthma",
    "Hypertension",
    "Migraine",
    "Cervical spondylosis",
    "Paralysis (brain hemorrhage)",
    "Jaundice",
    "Malaria",
    "Chicken pox",
    "Dengue",
    "Typhoid",
    "hepatitis A",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Hepatitis E",
    "Alcoholic hepatitis",
    "Tuberculosis",
    "Common Cold",
    "Pneumonia",
    "Dimorphic hemmorhoids(piles)",
    "Heart attack",
    "Varicose veins",
    "Hypothyroidism",
    "Hyperthyroidism",
    "Hypoglycemia",
    "Osteoarthristis",
    "Arthritis",
    "(vertigo) Paroymsal Positional Vertigo",
    "Acne",
    "Urinary tract infection",
    "Psoriasis",
    "Impetigo",
]

np.random.seed(42)
data_rows = []

# Generate 4,920 balanced training samples
for disease in diseases:
    for _ in range(120):
        row = np.zeros(len(symptoms), dtype=int)
        # Randomly select 3 to 6 active symptoms for each record
        active_indices = np.random.choice(
            len(symptoms), size=np.random.randint(3, 7), replace=False
        )
        row[active_indices] = 1
        data_rows.append(list(row) + [disease])

df_symptoms = pd.DataFrame(data_rows, columns=symptoms + ["prognosis"])
df_symptoms.to_csv("Training.csv", index=False)
print(
    f"✅ 'Training.csv' created successfully! ({len(df_symptoms)} rows × {len(df_symptoms.columns)} columns)"
)

# =====================================================================
# 2. CREATE DRUG & MEDICATION DATASET (Drug_Dataset.csv)
# =====================================================================
drug_records = [
    {
        "Drug_Name": "Clotrimazole Ointment / Ketoconazole Cream",
        "Reason": "Fungal infection",
        "Description": "Topical antifungal treatment to eliminate skin fungal spores and stop itching.",
    },
    {
        "Drug_Name": "Cetirizine 10mg / Loratadine 10mg",
        "Reason": "Allergy",
        "Description": "Antihistamine medication used for fast relief of itching, sneezing, and watery eyes.",
    },
    {
        "Drug_Name": "Omeprazole 20mg / Pantoprazole 40mg",
        "Reason": "GERD",
        "Description": "Proton pump inhibitor that significantly reduces acid secretion in the stomach.",
    },
    {
        "Drug_Name": "Ursodeoxycholic Acid 300mg",
        "Reason": "Chronic cholestasis",
        "Description": "Medication used to improve liver function and bile acid flow.",
    },
    {
        "Drug_Name": "Diphenhydramine / Hydrocortisone Cream",
        "Reason": "Drug Reaction",
        "Description": "Symptomatic treatment for adverse cutaneous drug eruptions and allergic rash.",
    },
    {
        "Drug_Name": "Ranitidine 150mg / Sucralfate Suspension",
        "Reason": "Peptic ulcer diseae",
        "Description": "Coats stomach ulcers and neutralizes stomach acid to reduce pain and allow healing.",
    },
    {
        "Drug_Name": "Antiretroviral Therapy (ART)",
        "Reason": "AIDS",
        "Description": "Combination viral suppression medications for immune deficiency management.",
    },
    {
        "Drug_Name": "Metformin 500mg / Glimepiride 2mg",
        "Reason": "Diabetes",
        "Description": "Oral hypoglycemic agents that regulate blood glucose concentrations.",
    },
    {
        "Drug_Name": "Oral Rehydration Salt (ORS) / Ondansetron",
        "Reason": "Gastroenteritis",
        "Description": "Restores fluid electrolyte balance and stops acute nausea and vomiting.",
    },
    {
        "Drug_Name": "Salbutamol Inhaler / Budesonide",
        "Reason": "Bronchial Asthma",
        "Description": "Bronchodilator and inhaled steroid for rapid relief of wheezing and airway constriction.",
    },
    {
        "Drug_Name": "Amlodipine 5mg / Telmisartan 40mg",
        "Reason": "Hypertension",
        "Description": "Antihypertensive agents used to lower systemic blood pressure.",
    },
    {
        "Drug_Name": "Sumatriptan 50mg / Naproxen 500mg",
        "Reason": "Migraine",
        "Description": "Selective serotonin receptor agonist for acute migraine pain and aura.",
    },
    {
        "Drug_Name": "ACN Gel (Topical) / Clindamycin Phosphate",
        "Reason": "Acne",
        "Description": "Antibacterial and retinoid topical treatment for acne spots and pimples.",
    },
    {
        "Drug_Name": "Nitrofurantoin 100mg / Ciprofloxacin 500mg",
        "Reason": "Urinary tract infection",
        "Description": "Targeted urinary anti-infective treatment for bladder and urinary symptoms.",
    },
    {
        "Drug_Name": "Clobetasol Propionate / Coal Tar Ointment",
        "Reason": "Psoriasis",
        "Description": "Topical steroid to reduce scaly skin lesions, inflammation, and flaking.",
    },
    {
        "Drug_Name": "Mupirocin Ointment / Oral Cephalexin",
        "Reason": "Impetigo",
        "Description": "Topical and systemic antibiotic coverage for crusted contagious skin sores.",
    },
    {
        "Drug_Name": "Cinnarizine 25mg / Betahistine 16mg",
        "Reason": "(vertigo) Paroymsal Positional Vertigo",
        "Description": "Vestibular suppressant used to alleviate spinning sensations and loss of balance.",
    },
]

df_drugs = pd.DataFrame(drug_records)
df_drugs.to_csv("Drug_Dataset.csv", index=False)
print(f"✅ 'Drug_Dataset.csv' created successfully! ({len(df_drugs)} records)")

print("\n🎉 Both dataset files are ready! You can now run 'streamlit run app.py'.")