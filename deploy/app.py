from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import io
from tensorflow.keras.models import load_model

app = FastAPI(title="ECG Feature-Based Prediction API")

# Load the trained Keras model
model = load_model("deploy/ecg_cnn_model.h5")
print("Model loaded successfully!")

# Label mapping
label_mapping = {0: "Normal", 1: "Abnormal"}

@app.post("/predict")
async def predict_ecg(file: UploadFile = File(...)):
    try:
        # Check that the uploaded file is a .npy file
        if not file.filename.endswith('.npy'):
            raise HTTPException(status_code=400, detail="File must be a .npy NumPy file")

        # Read the .npy file into a NumPy array
        contents = await file.read()
        input_array = np.load(io.BytesIO(contents))

        # Ensure the input shape is correct
        if input_array.ndim == 1:
            input_array = input_array.reshape(1, -1)
        elif input_array.ndim != 2 or input_array.shape[0] != 1:
            raise HTTPException(status_code=400, detail="Input array must have shape (1, N)")

        # Predict
        prediction_prob = model.predict(input_array)[0][0]

        predicted_class = int(prediction_prob > 0.5)
        result_label = label_mapping[predicted_class]

        confidence = prediction_prob if predicted_class == 1 else 1 - prediction_prob

        return {
            "prediction": result_label,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
