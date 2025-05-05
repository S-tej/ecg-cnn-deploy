import numpy as np
import requests

# Step 1: Create the example ECG feature input
# Example: [RR_interval, qrs_interval, pq_interval, st_interval, qt_interval]
input_features = np.array([[850, 100, 140, 110, 380]])  # shape (1, 5)

# Save to .npy file
npy_file_path = "input_features.npy"
np.save(npy_file_path, input_features)
print(f"✅ Saved test input to {npy_file_path}")

# Step 2: Send the .npy file to the FastAPI endpoint
# API endpoint (deployed)
url = "https://ecg-cnn-deploy-8.onrender.com/predict"

# Open and send the file as multipart/form-data
with open(npy_file_path, 'rb') as f:
    files = {'file': (npy_file_path, f, 'application/octet-stream')}
    response = requests.post(url, files=files)

# Step 3: Check and print the response
if response.status_code == 200:
    print("✅ Prediction result:")
    print(response.json())
else:
    print(f"❌ Error {response.status_code}: {response.text}")
