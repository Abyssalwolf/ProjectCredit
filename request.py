import requests

# Example input data (replace with your actual feature values)
input_data = {
    "LIMIT_BAL": 200000,
    "SEX": "1",  # Must match training encoding (e.g., "1" or "2")
    "EDUCATION": "2",
    "MARRIAGE": "1",
    "AGE": 30,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 50000,
    "BILL_AMT2": 48000,
    "BILL_AMT3": 46000,
    "BILL_AMT4": 44000,
    "BILL_AMT5": 42000,
    "BILL_AMT6": 40000,
    "PAY_AMT1": 2000,
    "PAY_AMT2": 2000,
    "PAY_AMT3": 2000,
    "PAY_AMT4": 2000,
    "PAY_AMT5": 2000,
    "PAY_AMT6": 2000
}

response = requests.post('http://localhost:5000/predict', json=input_data)
print(response.json())