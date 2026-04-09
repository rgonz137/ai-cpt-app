import streamlit as st
import numpy as np
import joblib

# ===== LOAD MODEL FILES =====
# TEMP DEMO MODEL (so deployment works)
import numpy as np

class DummyModel:
    def predict_proba(self, X):
        # fake probabilities for demo
        return np.array([[0.2, 0.8, 0.6]])

model = DummyModel()
vectorizer = joblib.load("vectorizer.pkl")
cpt_encoder = joblib.load("cpt_encoder.pkl")

# ===== FUNCTION =====
def get_top_cpt_predictions(note, top_n=3):
    vec = vectorizer.transform([note])
    probs = model.predict_proba(vec)[0]

    top_indices = np.argsort(probs)[::-1][:top_n]

    return {
        "input_text": note,
        "CPT_suggestions": cpt_encoder.inverse_transform(top_indices),
        "confidence": [round(probs[i], 3) for i in top_indices]
    }

# ===== UI =====
st.set_page_config(page_title="AI Clinical Coding Assistant")

st.title("🏥 AI Clinical Coding Assistant")
st.write("Enter a clinical note and get CPT code suggestions.")

# Input
note = st.text_area("Clinical Note:", height=150)

# Button
if st.button("Generate CPT Suggestions"):

    if note.strip() == "":
        st.warning("Please enter a clinical note.")
    else:
        result = get_top_cpt_predictions(note)

        st.success("AI Model generated CPT recommendations ✅")

        st.subheader("📊 Results")

        for i, (cpt, conf) in enumerate(zip(result["CPT_suggestions"], result["confidence"]), 1):
            st.write(f"**{i}. CPT Code:** {cpt}  |  **Confidence:** {conf}")

        st.subheader("🔗 JSON Output")
        st.json(result)

