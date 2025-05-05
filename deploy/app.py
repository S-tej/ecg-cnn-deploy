from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI(title="ECG Feature-Based Prediction API")

# Load the trained Keras model
model = load_model("deploy/ecg_cnn_model.h5")
print("Model loaded successfully!")

# Define the expected input features
class ECGFeatures(BaseModel):
    RR_interval: float
    qrs_interval: float
    pq_interval: float
    st_interval: float
    qt_interval: float

# Label mapping
label_mapping = {0: "Normal", 1: "Abnormal"}

@app.post("/predict")
async def predict_ecg(file: UploadFile = File(...)):  # Expecting a file
    contents = await file.read()

    # Assume the file is in CSV format, load it as needed
    # Example for a CSV file:
    import io
    import pandas as pd
    df = pd.read_csv(io.BytesIO(contents))

    # You could then extract features from the file and make predictions
    # Assuming the CSV has columns matching ECGFeatures:
    input_array = np.array([
        df['RR_interval'].values[0],  # Assuming a single row
        df['qrs_interval'].values[0],
        df['pq_interval'].values[0],
        df['st_interval'].values[0],
        df['qt_interval'].values[0]
    ]).reshape(1, -1)

    # Predict
    prediction_prob = model.predict(input_array)[0][0]
    predicted_class = int(prediction_prob > 0.5)
    result_label = label_mapping[predicted_class]

    confidence = prediction_prob if predicted_class == 1 else 1 - prediction_prob

    return {
        "prediction": result_label,
        "confidence": round(confidence * 100, 2)
    }
