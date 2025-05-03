from fastapi import FastAPI, UploadFile, File
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Load model once at startup
model = load_model("deploy/ecg_cnn_model.h5")
print("Model loaded successfully!")

@app.get("/")
def root():
    return {"message": "ECG CNN Model API is running!"}

@app.post("/evaluate")
async def evaluate_uploaded_data(
    x_file: UploadFile = File(...),
    y_file: UploadFile = File(...)
):
    # Load uploaded numpy files
    X_test = np.load(await x_file.read())
    y_test = np.load(await y_file.read())

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Predict
    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype(int)
    class_mapping = {0: "Normal", 1: "Abnormal"}
    predicted_labels = [class_mapping[cls[0]] for cls in predicted_classes]

    # Return first 10 predictions
    results = [
        {"sample": i + 1, "predicted_condition": label}
        for i, label in enumerate(predicted_labels[:10])
    ]

    return {
        "test_accuracy": f"{test_acc:.4f}",
        "samples": results
    }
