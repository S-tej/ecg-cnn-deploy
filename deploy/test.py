import requests

# URL of the FastAPI endpoint
url = "https://ecg-cnn-deploy-10.onrender.com/predict"

# Path to your local ECG CSV file
file_path = "deploy/xx.csv"

# Prepare the file for upload
with open(file_path, 'rb') as f:
    files = {'file': f}

    # Send the POST request to the FastAPI server with the file
    try:
        response = requests.post(url, files=files)
        
        # Check if the response status code is 200 (OK)
        response.raise_for_status()

        # Parse and display the response (prediction and confidence)
        result = response.json()
        print("Prediction:", result["prediction"])
        print("Confidence:", result["confidence"], "%")

    except requests.exceptions.HTTPError as e:
        print("HTTP error:", e)
        print("Response content:", response.text)
    except Exception as e:
        print("Request failed:", e)
