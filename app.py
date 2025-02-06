import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Import Langchain and Hugging Face Hub integration
from langchain import HuggingFaceHub
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API token from environment variable
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the Hugging Face Hub instance for LLaMA model
llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    huggingfacehub_api_token=api_token,
    model_kwargs={"temperature": 0.7, "max_length": 100}
)

# Load dataset
df = pd.read_csv("dataset.csv")

# Drop unnecessary columns
columns_to_drop = ['Patient_ID', 'Dietary_Restrictions', 'Allergies', 'Preferred_Cuisine',
                   'Blood_Pressure_mmHg', 'Glucose_mg/dL', 'Adherence_to_Diet_Plan', 'BMI']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
df = df.dropna().reset_index(drop=True)

# Label Encoding for categorical columns
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Diet_Recommendation'] = label_encoder.fit_transform(df['Diet_Recommendation'])

# One-Hot Encoding for categorical features
categorical_cols = ['Disease_Type', 'Severity', 'Physical_Activity_Level']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  
encoded_array = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# Preprocessing function for user input
def preprocess_input(data):
    """Preprocess user input for model prediction."""
    data['Gender'] = label_encoder.transform([data['Gender']])[0]  # Using the same label encoder for gender
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    missing_cols = set(df.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Handle missing columns
    input_df = input_df[df.drop(columns=['Diet_Recommendation']).columns]
    return input_df

# Train Random Forest Model (if not already trained)
X = df.drop(columns=['Diet_Recommendation'])   
y = df['Diet_Recommendation']  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save the model and scaler if needed
joblib.dump(model, "diet_model.pkl")
joblib.dump(scaler, "scaler.pkl")

def predict_diet_with_llama(user_input):
    """Generate diet recommendation using LLaMA model."""
    query = f"Provide a diet recommendation for a person with the following details: {user_input}"
    response = llm(query)
    return response

# Streamlit UI
st.title("Diet Recommendation System")

# User Inputs for Diet Prediction
age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", ['Male', 'Female'])
weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
daily_calories = st.number_input("Daily Caloric Intake", min_value=500, max_value=5000, value=2000)
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
exercise_hours = st.number_input("Weekly Exercise Hours", min_value=0, max_value=50, value=5)
nutrient_score = st.number_input("Dietary Nutrient Imbalance Score", min_value=0, max_value=100, value=50)
disease = st.selectbox("Disease Type", ['Diabetes', 'Hypertension', 'Obesity'])
severity = st.selectbox("Severity", ['Mild', 'Moderate', 'Severe'])
activity_level = st.selectbox("Physical Activity Level", ['Sedentary', 'Moderate', 'Active'])

# Prediction Button
if st.button("Predict Diet Recommendation"):
    user_data = {
        'Age': age,
        'Gender': gender,
        'Weight_kg': weight,
        'Height_cm': height,
        'Daily_Caloric_Intake': daily_calories,
        'Cholesterol_mg/dL': cholesterol,
        'Weekly_Exercise_Hours': exercise_hours,
        'Dietary_Nutrient_Imbalance_Score': nutrient_score,
        'Disease_Type': disease,
        'Severity': severity,
        'Physical_Activity_Level': activity_level
    }
    
    # Prepare the user input for prediction
    user_input = f"Age: {age}, Gender: {gender}, Weight: {weight}kg, Height: {height}cm, Daily Calories: {daily_calories}, Cholesterol: {cholesterol}mg/dL, Weekly Exercise: {exercise_hours} hours, Nutrient Imbalance: {nutrient_score}, Disease: {disease}, Severity: {severity}, Activity Level: {activity_level}"
    
    # Predict diet recommendation using LLaMA model
    result = predict_diet_with_llama(user_input)

    # Display the result
    st.success(f"Predicted Diet Recommendation: {result}")

