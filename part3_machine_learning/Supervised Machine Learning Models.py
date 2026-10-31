import sys
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Deep Learning & Explainability Toolkits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import shap

#Configure logging layout to append to your unified log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("pipeline.log", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Deep Learning with Explainability")
data_source = "ml_traffic_accident_proxy.csv"

logger.info("====================================================")
logger.info("LAUNCHING TASK 3: DEEP LEARNING WITH EXPLAINABILITY")
logger.info("====================================================")

if not os.path.exists(data_source):
    logger.error(f"Inbound Data Source Error: {data_source} missing. Execute Proxy accident-risk label.py first.")
    sys.exit(1)

df = pd.read_csv(data_source)
df['date_time'] = pd.to_datetime(df['date_time'])
df = df.sort_values('date_time').reset_index(drop=True)

#Select core features for sequential modeling
feature_cols = ['Hour_Sin', 'Hour_Cos', 'Scaled_Temp', 'Normalised_Clouds', 'Is_Weekend']
X_data = df[feature_cols].values
y_data = df['traffic_volume'].values

#Scale targets for neural network stability
scaler_y = MinMaxScaler()
y_data_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1))

#Construct sequential time-series windows (24-hour lookback window to predict the 25th hour)
lookback = 24
X_seq, y_seq = [], []
for i in range(len(df) - lookback):
    X_seq.append(X_data[i:(i + lookback)])
    y_seq.append(y_data_scaled[i + lookback])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

#Train/Test Split along sequence timelines
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
logger.info(f"Sequence matrices compiled. Train Shape: {X_train_dl.shape} | Test Shape: {X_test_dl.shape}")

logger.info("Initializing Architecture: Building Multi-Layer Recurrent LSTM Model")

#Construct the recurrent sequential network using modern Keras 3 Input objects
model = Sequential([
    tf.keras.layers.Input(shape=(lookback, len(feature_cols))), # Resolves the UserWarning cleanly
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1) # Linear output layer for continuous demand prediction
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

#Train network with compact epochs for rapid, clean notebook demonstration
logger.info("Starting LSTM Network Training Fit Loop...")
history = model.fit(
    X_train_dl, y_train_dl, 
    epochs=3, 
    batch_size=128, 
    validation_split=0.1, 
    verbose=1
)

#Evaluate metrics on unseen test sequences
lstm_loss, lstm_mae_scaled = model.evaluate(X_test_dl, y_test_dl, verbose=0)
#Re-scale metrics back to true vehicle dimensions
lstm_mae_vehicles = lstm_mae_scaled * (df['traffic_volume'].max() - df['traffic_volume'].min())

logger.info(f"LSTM Deep Learning Model Evaluation Complete -> Test Loss (MSE): {lstm_loss:.5f}")
logger.warning(f"LSTM Demand Prediction Performance Metric  -> True MAE: {lstm_mae_vehicles:.2f} vehicles")

logger.info("Initializing Explainability Stage: Building Comparable Tree Model for SHAP Evaluation")

#Flaten features into an aggregate mean-window layout to train a comparable tabular model
X_flat_train = np.mean(X_train_dl, axis=1)
X_flat_test = np.mean(X_test_dl, axis=1)
y_flat_train = y_train_dl.flatten()

#Train a comparable Random Forest Regressor
tree_explainer_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
tree_explainer_model.fit(X_flat_train, y_flat_train)

logger.info("Calculating Shapley Additive exPlanations (TreeSHAP) matrices...")
# Extract SHAP values
explainer = shap.TreeExplainer(tree_explainer_model)
shap_values = explainer.shap_values(X_flat_test[:500]) # Sample subset to execute quickly and cleanly

#Construct and export the formal SHAP Summary Plot to disk space
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_flat_test[:500], feature_names=feature_cols, show=False)
plt.title("SHAP Feature Importance Summary (Comparable Traffic Demand Model)", fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()

shap_plot_path = "shap_feature_importance.png"
plt.savefig(shap_plot_path, dpi=300)
plt.close()

logger.info(f"Successfully saved explainability figure: {shap_plot_path}")
logger.info("Task 3: Deep Learning with Explainability execution finalized successfully.")
logger.info("====================================================")