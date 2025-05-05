import requests

url = "https://ecg-cnn-deploy-10.onrender.com/predict"

# Input data matching the ECGFeatures schema
data = {
    "RR_interval": 0.85,
    "qrs_interval": 0.09,
    "pq_interval": 0.16,
    "st_interval": 0.12,
    "qt_interval": 0.38
}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raise an error if status != 200

    result = response.json()
    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"], "%")

except requests.exceptions.HTTPError as e:
    print("HTTP error:", e)
    print("Response content:", response.text)
except Exception as e:
    print("Request failed:", e)
