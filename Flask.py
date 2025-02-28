from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib
import google.generativeai as genai  # Corrected import
import os

app = Flask(__name__)

# Load your trained model and scalers
try:
    model = joblib.load("trained_model.pkl")
    x_scaler = joblib.load("x_scaler.pkl")
    y_scaler = joblib.load("y_scaler.pkl")
except Exception as e:
    print(f"Error loading model or scalers: {e}")
    model, x_scaler, y_scaler = None, None, None

# Configure Google Generative AI with your API key securely
genai.configure(api_key=os.getenv("AIzaSyCDSR_hKtxnySAodqysXk-njSXcMTGIt2s"))

# Define the expected input feature names and output names
FEATURE_NAMES = [
    "Income", "Rent", "Insurance", "Groceries", "Transport", "Eating_Out",
    "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous",
    "Desired_Savings", "Disposable_Income"
]

OUTPUT_NAMES = [
    "Potential_Savings_Groceries", "Potential_Savings_Transport", "Potential_Savings_Eating_Out",
    "Potential_Savings_Entertainment", "Potential_Savings_Utilities", "Potential_Savings_Healthcare",
    "Potential_Savings_Education"
]

# Helper function using Google's Gemini AI API
def get_google_response(prompt):
    try:
        response = genai.generate_content(prompt)
        return response.text if response else "No response received."
    except Exception as e:
        return f"Error contacting Google GenAI API: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            input_features = []
            for feature in FEATURE_NAMES:
                value = request.form.get(feature, "").strip()
                if not value.replace('.', '', 1).isdigit():  # Check for valid numeric values
                    return render_template("index.html", error=f"Invalid input for {feature}. Please enter a numeric value.", features=FEATURE_NAMES)
                input_features.append(float(value))
            
            input_array = np.array(input_features).reshape(1, -1)
            print(f"Original input shape: {input_array.shape}")  # Debugging
            input_scaled = x_scaler.transform(input_array)

            if input_scaled.shape[1] != model.input_shape[1]:
                return render_template("index.html", error=f"Input shape mismatch. Expected {model.input_shape[1]}, but got {input_scaled.shape[1]}", features=FEATURE_NAMES)

            prediction_scaled = model.predict(input_scaled)
            prediction = y_scaler.inverse_transform(prediction_scaled).flatten().tolist()
            results = dict(zip(OUTPUT_NAMES, prediction))
            return render_template("result.html", results=results)
        except ValueError as e:
            return render_template("index.html", error=f"Input error: {str(e)}", features=FEATURE_NAMES)
    
    return render_template("index.html", features=FEATURE_NAMES)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    conversation = []
    response_text = ""
    
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if user_input:
            conversation.append(("User", user_input))
            prompt = (
                "Based on my current financial inputs and predictions, "
                "what steps can I take to improve my financial status? "
                f"Here is my question: {user_input}"
            )
            google_response = get_google_response(prompt)
            conversation.append(("Google GenAI", google_response))
            response_text = google_response
    
    return render_template("chat.html", conversation=conversation, response_text=response_text)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
