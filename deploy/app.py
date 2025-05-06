from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI(title="ECG Feature-Based Prediction API")

# Load the trained Keras model
model = load_model("deploy/ecg_cnn_model.h5")
print("Model loaded successfully!")

# Define expected input features
class ECGFeatures(BaseModel):
    RR_interval: float
    qrs_interval: float
    pq_interval: float
    st_interval: float
    qt_interval: float

# Label mapping
label_mapping = {1: "Normal", 0: "Abnormal"}

# Example: max values (adjust if you used scaling during training)
# If you **did not scale** during training, REMOVE this scaling step.
max_values = np.array([1.0, 1.0, 1.0, 0.999999, 0.999999])  # adjust if needed

@app.post("/predict")
async def predict_ecg(features: ECGFeatures):
    # Convert input to NumPy array
    raw_input = np.array([
        [features.RR_interval, 0.0],
        [features.qrs_interval, 0.0],
        [features.pq_interval, 0.0],
        [features.st_interval, 0.0],
        [features.qt_interval, 0.0]
    ])
    print("Raw input:", raw_input)

    # Optional: apply scaling if you used it during training
    scaled_input = raw_input.copy()
    scaled_input[:, 0] = scaled_input[:, 0] / max_values  # only first column
    print("Scaled input:", scaled_input)

    # Prepare input shape (1, 5, 2)
    input_array = scaled_input.reshape(1, 5, 2)
    print("Input array shape:", input_array.shape)

    # Get model prediction
    prediction_output = model.predict(input_array)
    print("Raw model output:", prediction_output)

    # Single sigmoid output
    prediction_prob = float(prediction_output[0][0])
    predicted_class = int(prediction_prob > 0.5)
    confidence = prediction_prob if predicted_class == 1 else 1 - prediction_prob

    result_label = label_mapping.get(predicted_class, "Unknown")
    final = "Abnormal"
    if (result_label=="Abnormal"):
        final = "Normal"

    return {
        "prediction": final,
        "confidence": round(confidence * 100, 2)
    }
