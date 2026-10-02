import datetime
import jwt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SECRET_KEY = "healthcare_secret_key_jwt"


# =====================================================================
# 1. JWT AUTHENTICATION ENGINE
# =====================================================================
def generate_token(username: str, role: str) -> str:
    payload = {
        "username": username,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def verify_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return "Token Expired"
    except jwt.InvalidTokenError:
        return "Invalid Token"


# =====================================================================
# 2. CACHED DISEASE PREDICTION ENGINE
# =====================================================================
def train_disease_model(symptom_df: pd.DataFrame):
    X = symptom_df.drop(columns=["prognosis"])
    y = symptom_df["prognosis"]

    model = RandomForestClassifier(
        n_estimators=30, max_depth=15, n_jobs=1, random_state=42
    )
    model.fit(X, y)

    return model, list(X.columns)


def predict_prognosis(model, symptom_cols: list, selected_symptoms: list):
    input_vector = np.zeros(len(symptom_cols))

    for symptom in selected_symptoms:
        if symptom in symptom_cols:
            idx = symptom_cols.index(symptom)
            input_vector[idx] = 1

    prediction = model.predict([input_vector])[0]
    probabilities = model.predict_proba([input_vector])[0]
    confidence = max(probabilities) * 100

    return prediction, round(confidence, 2)


# =====================================================================
# 3. CONTENT-BASED DRUG RECOMMENDATION ENGINE
# =====================================================================
class ContentBasedRecommender:

    def __init__(self, drug_df: pd.DataFrame):
        self.drug_df = drug_df

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        combined_text = (
            self.drug_df["Reason"].fillna("").astype(str)
            + " "
            + self.drug_df["Description"].fillna("").astype(str)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(combined_text)

    def recommend(self, disease_name: str, top_n: int = 5) -> pd.DataFrame:
        direct_matches = self.drug_df[
            self.drug_df["Reason"].str.contains(disease_name, case=False, na=False)
        ]

        if not direct_matches.empty:
            cols_to_show = [
                col
                for col in ["Drug_Name", "Reason", "Description"]
                if col in direct_matches.columns
            ]
            return direct_matches[cols_to_show].head(top_n)

        query_vec = self.vectorizer.transform([disease_name])
        similarity_scores = cosine_similarity(
            query_vec, self.tfidf_matrix
        ).flatten()

        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        recommended_df = self.drug_df.iloc[top_indices]

        cols_to_show = [
            col
            for col in ["Drug_Name", "Reason", "Description"]
            if col in recommended_df.columns
        ]
        return recommended_df[cols_to_show]