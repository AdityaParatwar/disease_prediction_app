import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
svc = pickle.load(open("svc.pkl", "rb"))

# Load and clean CSV files
description = pd.read_csv("description.csv")
precautions = pd.read_csv("precautions_df.csv")
medications = pd.read_csv("medications.csv")
workout = pd.read_csv("workout_df.csv")
diets = pd.read_csv("diets.csv")

# Normalize column names for consistency
description.columns = description.columns.str.strip().str.capitalize()
precautions.columns = precautions.columns.str.strip().str.capitalize()
medications.columns = medications.columns.str.strip().str.replace(' ', '_').str.capitalize()
workout.columns = workout.columns.str.strip().str.capitalize()
diets.columns = diets.columns.str.strip().str.capitalize()

# Symptom list (should match training)
symptom_list = [ 'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
    'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
    'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety',
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
    'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin',
    'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
    'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
    'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus',
    'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
    'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
    'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
    'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
    'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine',
    'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability',
    'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation',
    'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history',
    'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
    'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
    'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
    'blister', 'red_sore_around_nose', 'yellow_crust_ooze' ]

# Streamlit UI setup
st.set_page_config(page_title="AI Medical Diagnosis", layout="centered", page_icon="🧬")
st.title("🧠 AI-Based Medical Diagnosis Assistant")
st.markdown("### 🔍 Select Symptoms")
selected_symptoms = st.multiselect("Choose from the list below:", sorted(symptom_list))

# Predict function
def predict(symptoms):
    input_vector = [1 if s in symptoms else 0 for s in symptom_list]
    return svc.predict([input_vector])[0]

# Show detailed disease info
def show_info(disease):
    # Description
    desc = description[description['Disease'] == disease]['Description'].values
    if desc.size > 0:
        st.markdown(f"### 🩺 Disease Description\n{desc[0]}")

    # Precautions
    prec = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
    if prec.size > 0:
        st.markdown("### 🚑 Precautions")
        for item in prec[0]:
            if pd.notna(item):
                st.write(f"- {item}")

    # Medications
    meds_row = medications[medications['Disease'] == disease]
    if not meds_row.empty:
        st.markdown("### 💊 Medications")
        for col in meds_row.columns:
            if "Medication" in col:
                med = meds_row[col].values[0]
                if pd.notna(med):
                    st.text(f"{med}")

    # Workout Tips
    tips = workout[workout['Disease'] == disease]['Workout'].values
    if tips.size > 0:
        st.markdown("### 🏃 Workout Tips")
        for tip in tips:
            if pd.notna(tip):
                st.write(f"- {tip}")

    # Diets
    diet_row = diets[diets['Disease'] == disease]['Diet'].values
    if diet_row.size > 0:
        st.markdown("### 🍎 Recommended Diet")
        try:
            items = eval(diet_row[0])  # Safely parse list string
            for food in items:
                st.write(f"- {food}")
        except:
            st.write(diet_row[0])

# Predict button
if st.button("🔮 Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        result = predict(selected_symptoms)
        st.success(f" Predicted Disease: **{result}**")
        show_info(result)
