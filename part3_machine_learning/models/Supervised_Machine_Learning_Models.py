import sys
import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

#=====================================================================
#1. CENTRALIZED LOGGER CONFIGURATION
#=====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("pipeline.log", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Supervised Machine Learning Models")
data_source = "ml_traffic_accident_proxy.csv"

logger.info("====================================================")
logger.info("LAUNCHING TASK 1: SUPERVISED MACHINE LEARNING MODELS")
logger.info("====================================================")

#--------------------------------------------------------------------
#2. DATA INGESTION & FEATURE ENGINEERING MAPS
#--------------------------------------------------------------------
if not os.path.exists(data_source):
    logger.error(f"Inbound Data Error: {data_source} missing. Execute Proxy_accident-risk_label.py first.")
    sys.exit(1)

df = pd.read_csv(data_source)
logger.info(f"Successfully loaded dataset matrix. Shape: {df.shape}")

#Convert 'holiday' to a strict binary numeric flag indicator (1 if holiday exists, else 0)
df['Is_Holiday'] = np.where(df['holiday'] != 'None', 1, 0)

#Apply cyclical encoding of day of week to sit alongside hour cyclical wave encodings
df['Day_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7.0)
df['Day_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7.0)

#Define common unified well-engineered feature set array
weather_features = [col for col in df.columns if col.startswith('Weather_') and col != 'Weather_Unknown']
feature_cols = [
    'Hour', 'Day_of_Week', 'Is_Weekend', 'Is_Holiday',
    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
    'Scaled_Temp', 'Normalised_Clouds'
] + weather_features

X = df[feature_cols]
y_reg = df['traffic_volume']
y_clf = df['high_risk']

logger.info(f"Feature set locked. Total features utilized: {len(feature_cols)}")

#=====================================================================
#3. REGRESSION MODEL PIPELINE (PREDICTING TRAFFIC VOLUME)
#=====================================================================
logger.info("Initializing Stage: Regression Modeling (Target: traffic_volume)")

#Split Regression Arrays
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

#Scale Feature Distributions for Linear Stability
scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

#Algorithm 1: Baseline Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_r_scaled, y_train_r)
lr_preds = lr_model.predict(X_test_r_scaled)

#Algorithm 2: Random Forest Ensemble Regressor
rf_reg = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_r, y_train_r) # Tree algorithms do not require manual scaling
rf_reg_preds = rf_reg.predict(X_test_r)

#Evaluate Regression Performance Metrics
mae_lr, r2_lr = mean_absolute_error(y_test_r, lr_preds), r2_score(y_test_r, lr_preds)
mae_rf, r2_rf = mean_absolute_error(y_test_r, rf_reg_preds), r2_score(y_test_r, rf_reg_preds)

logger.info(f"Regression Baseline (Linear Regression) -> MAE: {mae_lr:.2f}, R²: {r2_lr:.4f}")
logger.warning(f"Regression Ensemble (Random Forest) -> MAE: {mae_rf:.2f}, R²: {r2_rf:.4f}")

#=====================================================================
#4. CLASSIFICATION MODEL PIPELINE (ESTIMATING ACCIDENT LIKELIHOOD)
#=====================================================================
logger.info("Initializing Stage: Classification Modeling (Target: high_risk proxy)")

#Split Classification Arrays
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

#Scale Feature Distributions
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

#Algorithm 1: Baseline Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_c_scaled, y_train_c)
log_preds = log_reg.predict(X_test_c_scaled)
log_probs = log_reg.predict_proba(X_test_c_scaled)[:, 1]

#Algorithm 2: Random Forest Ensemble Classifier
rf_clf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
rf_clf.fit(X_train_c, y_train_c)
rf_clf_preds = rf_clf.predict(X_test_c)
rf_clf_probs = rf_clf.predict_proba(X_test_c)[:, 1]

#Evaluate Classification Performance Metrics
metrics = {
    "Logistic Regression": [
        accuracy_score(y_test_c, log_preds), precision_score(y_test_c, log_preds),
        recall_score(y_test_c, log_preds), f1_score(y_test_c, log_preds), roc_auc_score(y_test_c, log_probs)
    ],
    "Random Forest Classifier": [
        accuracy_score(y_test_c, rf_clf_preds), precision_score(y_test_c, rf_clf_preds),
        recall_score(y_test_c, rf_clf_preds), f1_score(y_test_c, rf_clf_preds), roc_auc_score(y_test_c, rf_clf_probs)
    ]
}

for model_name, m_list in metrics.items():
    logger.info(
        f"{model_name} Outputs -> Accuracy: {m_list[0]:.4f} | Precision: {m_list[1]:.4f} | "
        f"Recall: {m_list[2]:.4f} | F1-score: {m_list[3]:.4f} | ROC AUC: {m_list[4]:.4f}"
    )

logger.info("====================================================")
logger.info("SUPERVISED MODELING EVALUATION PHASE MASTER COMPLETED")
logger.info("====================================================")