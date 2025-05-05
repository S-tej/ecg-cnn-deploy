from fastapi import FastAPI
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
async def predict_ecg(features: ECGFeatures):
    # Convert input features to NumPy array and reshape
    input_array = np.array([
        features.RR_interval,
        features.qrs_interval,
        features.pq_interval,
        features.st_interval,
        features.qt_interval
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
