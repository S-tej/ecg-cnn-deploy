import requests

url = "https://ecg-cnn-deploy-8.onrender.com/predict"

data = {
    "RR_interval": 0.85,
    "qrs_interval": 0.09,
    "pq_interval": 0.16,
    "st_interval": 0.12,
    "qt_interval": 0.38
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.json())