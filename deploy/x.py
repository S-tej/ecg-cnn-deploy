import numpy as np
from tensorflow.keras.models import load_model

model = load_model("deploy/ecg_cnn_model.h5")
print("Model loaded!")

# Example crafted input for Normal prediction
raw_input = np.array([
    [0.9, 0.0],   # RR_interval
    [0.09, 0.0],  # qrs_interval
    [0.16, 0.0],  # pq_interval
    [0.14, 0.0],  # st_interval
    [0.4, 0.0]    # qt_interval
]).reshape(1, 5, 2)

prediction_output = model.predict(raw_input)
print("Raw model output:", prediction_output)
