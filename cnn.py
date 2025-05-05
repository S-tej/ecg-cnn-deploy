import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Load preprocessed data
X = np.load("preprocessed_datasets/X.npy")
y = np.load("preprocessed_datasets/y.npy")

# Check if X needs reshaping
if X.shape[1] == 1:
    X = np.transpose(X, (0, 2, 1))  # (samples, time steps, channels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Compute class weights (for imbalance)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Build the improved CNN
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Conv1D(filters=64, kernel_size=3, activation="relu"),
    BatchNormalization(),
    GlobalMaxPooling1D(),

    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# Compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC()])

# Train with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=32,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)

# Evaluate
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {round(test_acc * 100, 2)}%, AUC: {round(test_auc, 4)}")

# Save
model.save("ecg_cnn_model.h5")
print("Model saved successfully!")
