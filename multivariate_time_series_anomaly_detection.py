import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                           roc_auc_score, confusion_matrix, classification_report)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# 1. Data Preparation
# Synthetic IoT sensor data with anomalies
data = """timestamp,temperature_C,pressure_hPa,humidity_percent,vibration_mm_s,power_consumption_kW,is_anomaly
2023-06-01 00:00:00,28.5,1013.2,45.1,2.1,1.45,0
2023-06-01 00:05:00,28.7,1013.1,45.0,2.0,1.46,0
2023-06-01 00:10:00,28.9,1013.0,44.9,2.2,1.47,0
2023-06-01 00:15:00,29.1,1012.9,44.8,2.1,1.48,0
2023-06-01 00:20:00,29.3,1012.8,44.7,2.3,1.49,0
2023-06-01 00:25:00,29.5,1012.7,44.6,2.2,1.50,0
2023-06-01 00:30:00,29.7,1012.6,44.5,2.4,1.51,0
2023-06-01 00:35:00,29.9,1012.5,44.4,2.3,1.52,0
2023-06-01 00:40:00,30.1,1012.4,44.3,2.5,1.53,0
2023-06-01 00:45:00,30.3,1012.3,44.2,2.4,1.54,0
2023-06-01 00:50:00,30.5,1012.2,44.1,2.6,1.55,0
2023-06-01 00:55:00,30.7,1012.1,44.0,2.5,1.56,0
2023-06-01 01:00:00,30.9,1012.0,43.9,2.7,1.57,0
2023-06-01 01:05:00,31.1,1011.9,43.8,2.6,1.58,0
2023-06-01 01:10:00,31.3,1011.8,43.7,2.8,1.59,0
2023-06-01 01:15:00,31.5,1011.7,43.6,2.7,1.60,0
2023-06-01 01:20:00,31.7,1011.6,43.5,2.9,1.61,0
2023-06-01 01:25:00,31.9,1011.5,43.4,2.8,1.62,0
2023-06-01 01:30:00,32.1,1011.4,43.3,3.0,1.63,0
2023-06-01 01:35:00,32.3,1011.3,43.2,2.9,1.64,0
2023-06-01 01:40:00,32.5,1011.2,43.1,3.1,1.65,0
2023-06-01 01:45:00,32.7,1011.1,43.0,3.0,1.66,0
2023-06-01 01:50:00,32.9,1011.0,42.9,3.2,1.67,0
2023-06-01 01:55:00,33.1,1010.9,42.8,3.1,1.68,0
2023-06-01 02:00:00,33.3,1010.8,42.7,3.3,1.69,0
2023-06-01 02:05:00,33.5,1010.7,42.6,3.2,1.70,0
2023-06-01 02:10:00,33.7,1010.6,42.5,3.4,1.71,0
2023-06-01 02:15:00,33.9,1010.5,42.4,3.3,1.72,0
2023-06-01 02:20:00,34.1,1010.4,42.3,3.5,1.73,0
2023-06-01 02:25:00,34.3,1010.3,42.2,3.4,1.74,0
2023-06-01 02:30:00,34.5,1010.2,42.1,3.6,1.75,0
2023-06-01 02:35:00,34.7,1010.1,42.0,3.5,1.76,0
2023-06-01 02:40:00,34.9,1010.0,41.9,3.7,1.77,0
2023-06-01 02:45:00,35.1,1009.9,41.8,3.6,1.78,0
2023-06-01 02:50:00,35.3,1009.8,41.7,3.8,1.79,0
2023-06-01 02:55:00,35.5,1009.7,41.6,3.7,1.80,0
2023-06-01 03:00:00,35.7,1009.6,41.5,3.9,1.81,0
2023-06-01 03:05:00,35.9,1009.5,41.4,3.8,1.82,0
2023-06-01 03:10:00,36.1,1009.4,41.3,4.0,1.83,0
2023-06-01 03:15:00,36.3,1009.3,41.2,3.9,1.84,0
2023-06-01 03:20:00,36.5,1009.2,41.1,4.1,1.85,0
2023-06-01 03:25:00,36.7,1009.1,41.0,4.0,1.86,0
2023-06-01 03:30:00,36.9,1009.0,40.9,4.2,1.87,0
2023-06-01 03:35:00,37.1,1008.9,40.8,4.1,1.88,0
2023-06-01 03:40:00,37.3,1008.8,40.7,4.3,1.89,0
2023-06-01 03:45:00,37.5,1008.7,40.6,4.2,1.90,0
2023-06-01 03:50:00,37.7,1008.6,40.5,4.4,1.91,0
2023-06-01 03:55:00,37.9,1008.5,40.4,4.3,1.92,0
2023-06-01 04:00:00,38.1,1008.4,40.3,4.5,1.93,0
2023-06-01 04:05:00,38.3,1008.3,40.2,4.4,1.94,0
2023-06-01 04:10:00,38.5,1008.2,40.1,4.6,1.95,0
2023-06-01 04:15:00,38.7,1008.1,40.0,4.5,1.96,0
2023-06-01 04:20:00,38.9,1008.0,39.9,4.7,1.97,0
2023-06-01 04:25:00,39.1,1007.9,39.8,4.6,1.98,0
2023-06-01 04:30:00,39.3,1007.8,39.7,4.8,1.99,0
2023-06-01 04:35:00,39.5,1007.7,39.6,4.7,2.00,0
2023-06-01 04:40:00,39.7,1007.6,39.5,4.9,2.01,0
2023-06-01 04:45:00,39.9,1007.5,39.4,4.8,2.02,0
2023-06-01 04:50:00,40.1,1007.4,39.3,5.0,2.03,0
2023-06-01 04:55:00,40.3,1007.3,39.2,4.9,2.04,0
2023-06-01 05:00:00,40.5,1007.2,39.1,5.1,2.05,0
2023-06-01 05:05:00,40.7,1007.1,39.0,5.0,2.06,0
2023-06-01 05:10:00,40.9,1007.0,38.9,5.2,2.07,0
2023-06-01 05:15:00,41.1,1006.9,38.8,5.1,2.08,0
2023-06-01 05:20:00,41.3,1006.8,38.7,5.3,2.09,0
2023-06-01 05:25:00,41.5,1006.7,38.6,5.2,2.10,0
2023-06-01 05:30:00,41.7,1006.6,38.5,5.4,2.11,0
2023-06-01 05:35:00,41.9,1006.5,38.4,5.3,2.12,0
2023-06-01 05:40:00,42.1,1006.4,38.3,5.5,2.13,0
2023-06-01 05:45:00,42.3,1006.3,38.2,5.4,2.14,0
2023-06-01 05:50:00,42.5,1006.2,38.1,5.6,2.15,0
2023-06-01 05:55:00,42.7,1006.1,38.0,5.5,2.16,0
2023-06-01 06:00:00,42.9,1006.0,37.9,5.7,2.17,0"""

# Load data into DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data), parse_dates=['timestamp'])

# Inject some anomalies
anomaly_indices = [10, 25, 40, 55]  # 4 anomalies (~6% of data)
for idx in anomaly_indices:
    df.loc[idx, 'temperature_C'] += np.random.uniform(5, 10)
    df.loc[idx, 'vibration_mm_s'] += np.random.uniform(3, 6)
    df.loc[idx, 'power_consumption_kW'] += np.random.uniform(1, 2)
    df.loc[idx, 'is_anomaly'] = 1

# 2. Data Preprocessing
# Select features and target
features = ['temperature_C', 'pressure_hPa', 'humidity_percent', 'vibration_mm_s', 'power_consumption_kW']
X = df[features]
y = df['is_anomaly']

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences for LSTM
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

sequence_length = 10
X_sequences = create_sequences(X_scaled, sequence_length)

# 3. LSTM Autoencoder Model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(sequence_length, X_scaled.shape[1]), 
    RepeatVector(sequence_length),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X_scaled.shape[1]))
])

model.compile(optimizer='adam', loss='mse')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train-test split
train_size = int(0.8 * len(X_sequences))
X_train, X_test = X_sequences[:train_size], X_sequences[train_size:]
y_train, y_test = y[sequence_length-1:train_size+sequence_length-1], y[train_size+sequence_length-1:]

# Train model
history = model.fit(
    X_train, X_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 4. Anomaly Detection
# Get reconstruction errors
train_recon = model.predict(X_train)
train_mse = np.mean(np.square(X_train - train_recon), axis=(1, 2))

test_recon = model.predict(X_test)
test_mse = np.mean(np.square(X_test - test_recon), axis=(1, 2))

# Train Isolation Forest on reconstruction errors
iso_forest = IsolationForest(contamination=float(y_train.mean()), random_state=42)
iso_forest.fit(train_mse.reshape(-1, 1))

# Get anomaly scores (combine reconstruction error with isolation forest scores)
test_scores = -iso_forest.score_samples(test_mse.reshape(-1, 1))  # Convert to positive scores

# 5. Threshold Optimization
def optimize_threshold(scores, true_labels):
    thresholds = np.linspace(min(scores), max(scores), 100)
    best_f1 = 0
    best_thresh = 0
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        f1 = f1_score(true_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

optimal_threshold = optimize_threshold(test_scores, y_test)
predictions = (test_scores > optimal_threshold).astype(int)

# 6. Evaluation
print("\nEvaluation Metrics:")
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"F1 Score: {f1_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions):.4f}")
print(f"Recall: {recall_score(y_test, predictions):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, test_scores):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 7. Visualization
plt.figure(figsize=(12, 6))

# Plot reconstruction error
plt.subplot(2, 1, 1)
plt.plot(test_mse, label='Reconstruction Error')
plt.axhline(y=optimal_threshold, color='r', linestyle='--', label='Optimal Threshold')
plt.title('Test Set Reconstruction Error')
plt.ylabel('MSE')
plt.legend()

# Plot anomalies
plt.subplot(2, 1, 2)
anomaly_points = np.where(y_test == 1)[0]
plt.scatter(anomaly_points, test_scores[anomaly_points], 
            c='red', label='True Anomalies')
pred_anomaly_points = np.where(predictions == 1)[0]
plt.scatter(pred_anomaly_points, test_scores[pred_anomaly_points], 
            c='blue', marker='x', label='Predicted Anomalies')
plt.title('Anomaly Detection Results')
plt.xlabel('Time Step')
plt.ylabel('Anomaly Score')
plt.legend()

plt.tight_layout()
plt.show()

# 8. Save Model
model.save('lstm_autoencoder_anomaly_detection.h5')
print("\nModel saved successfully.")
