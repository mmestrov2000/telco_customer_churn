import requests

payload = {
    "SeniorCitizen": 0,
    "tenure": 34,
    "MonthlyCharges": 56.0,
    "TotalCharges": 1889.0,
    "Partner_Yes": 0,
    "Dependents_Yes": 0,
    "MultipleLines_No phone service": 0,
    "MultipleLines_Yes": 0,
    "InternetService_Fiber optic": 0,
    "InternetService_No": 0,
    "OnlineSecurity_No internet service": 0,
    "OnlineSecurity_Yes": 1,
    "OnlineBackup_No internet service": 0,
    "OnlineBackup_Yes": 0,
    "DeviceProtection_No internet service": 0,
    "DeviceProtection_Yes": 1,
    "TechSupport_No internet service": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_No internet service": 0,
    "StreamingTV_Yes": 0,
    "StreamingMovies_No internet service": 0,
    "StreamingMovies_Yes": 0,
    "Contract_One year": 1,
    "Contract_Two year": 0,
    "PaperlessBilling_Yes": 0,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 0,
    "PaymentMethod_Mailed check": 1
}

url = "http://51.20.1.109:8000/predict"  # Replace with your public IP

res = requests.post(url, json=payload)
print("Churn Probability:", res.json())