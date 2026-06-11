import gradio as gr
import joblib
import numpy as np
import pandas as pd

# 1. Load the trained model and scaler
# These files must be uploaded to the Hugging Face Space!
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Define the prediction function
def predict_churn(age, gender, tenure, usage_frequency, support_calls, payment_delay, 
                  subscription_type, contract_length, total_spend, last_interaction):
    
    # --- Preprocessing: Replicating the Label Encoding ---
    # We must convert the user's friendly text into the numbers the model was trained on.
    
    # Gender: Female -> 0, Male -> 1
    gender_enc = 0 if gender == "Female" else 1
    
    # Subscription Type: Basic -> 0, Premium -> 1, Standard -> 2
    if subscription_type == "Basic":
        sub_enc = 0
    elif subscription_type == "Premium":
        sub_enc = 1
    else: # Standard
        sub_enc = 2
        
    # Contract Length: Annual -> 0, Monthly -> 1, Quarterly -> 2
    if contract_length == "Annual":
        contract_enc = 0
    elif contract_length == "Monthly":
        contract_enc = 1
    else: # Quarterly
        contract_enc = 2
    
    # Create the feature array (Order must be exactly same as training!)
    features = np.array([[age, gender_enc, tenure, usage_frequency, support_calls, 
                          payment_delay, sub_enc, contract_enc, total_spend, last_interaction]])
    
    # Scale the features using the loaded scaler
    features_scaled = scaler.transform(features)
    
    # --- Prediction ---
    prediction = model.predict(features_scaled)
    
    # Return user-friendly result
    if prediction[0] == 1:
        return "⚠️ High Risk: Customer is predicted to CHURN (leave)"
    else:
        return "✅ Low Risk: Customer is predicted to STAY (No Churn)"

# 3. Create the Gradio Interface
interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="1. Age"),
        gr.Radio(["Female", "Male"], label="2. Gender"),
        gr.Number(label="3. Tenure (Months)"),
        gr.Number(label="4. Usage Frequency"),
        gr.Number(label="5. Support Calls"),
        gr.Number(label="6. Payment Delay (Days)"),
        gr.Radio(["Basic", "Standard", "Premium"], label="7. Subscription Type"),
        gr.Radio(["Annual", "Monthly", "Quarterly"], label="8. Contract Length"),
        gr.Number(label="9. Total Spend"),
        gr.Number(label="10. Last Interaction (Days ago)")
    ],
    outputs="text",
    title="KNN Customer Churn Predictor (Your Project)",
    description="A K-Nearest Neighbors model predicting customer retention based on key features."
)

# 4. Launch the app (This only runs on a local machine, Hugging Face runs it differently)
if __name__ == "__main__":
    interface.launch()