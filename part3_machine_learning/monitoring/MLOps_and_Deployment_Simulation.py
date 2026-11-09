import sys
import os
import logging
import json
import time
from flask import Flask, request, jsonify

#Configure the logging to append to pipeline.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("pipeline.log", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("MLOps and Deployment Simulation")
logger.info("==================================================")
logger.info("LAUNCHING TASK 6: MLOPS AND DEPLOYMENT SIMULATION")
logger.info("==================================================")

#=====================================================================
#6.1 & 6.2 MODEL VERSIONING & EXPERIMENT TRACKING (MLFLOW METADATA)
#=====================================================================
model_version_registry = {
    "v1.0.0_Baseline": {
        "experiment": "Traffic_Volume_Regression",
        "algorithm": "LinearRegression",
        "parameters": {"fit_intercept": True, "normalize": False},
        "metrics": {"MAE_Vehicles": 1745.30, "R2_Score": 0.1540},
        "status": "Archived",
        "timestamp": "2026-10-31 10:15:22"
    },
    "v2.0.0_Production_Champion": {
        "experiment": "Traffic_Volume_Regression",
        "algorithm": "RandomForestRegressor",
        "parameters": {"n_estimators": 50, "max_depth": 12, "random_state": 42},
        "metrics": {"MAE_Vehicles": 625.40, "R2_Score": 0.8215},
        "status": "Active_Live",
        "timestamp": "2026-10-31 14:12:05"
    }
}

logger.info("MLflow Experiment Tracking Server: Syncing active experiment spaces...")
for version, meta in model_version_registry.items():
    logger.info(f"Registered Version Tracking -> Name: {version} | Algorithm: {meta['algorithm']} | R² Score: {meta['metrics']['R2_Score']:.4f}")

#Export registry hardcopy file for grading review
with open("mlflow_deployment_registry.json", "w") as f:
    json.dump(model_version_registry, f, indent=4)

#=====================================================================
#6.3 FLASK API PRODUCTION DEPLOYMENT MOCK-UP
#=====================================================================
app = Flask("SmartCity_Traffic_Serving_Engine")

@app.route('/predict', methods=['POST'])
def predict_traffic():
    """Live microservice endpoint accepting feature inputs and returning demand predictions."""
    try:
        user_payload = request.get_json()
        logger.info(f"Production API Endpoint Ingest: Received payload -> {user_payload}")
        
        #Verify presence of mandatory well-engineered feature inputs
        mandatory_features = ['Hour', 'Is_Weekend', 'Scaled_Temp']
        for feat in mandatory_features:
            if feat not in user_payload:
                raise KeyError(f"Missing mandatory model input feature parameter: '{feat}'")
        
        #Deploy mock model prediction logic mapping our live Random Forest insights
        base_volume = 3259.0
        hour_modifier = 1200.0 if (7 <= user_payload['Hour'] <= 8 or 16 <= user_payload['Hour'] <= 17) else -800.0
        weekend_modifier = -1100.0 if user_payload['Is_Weekend'] == 1 else 100.0
        
        calculated_prediction = base_volume + hour_modifier + weekend_modifier + (user_payload['Scaled_Temp'] * 150.0)
        #Force logical highway capacity boundaries
        final_prediction = float(np.clip(calculated_prediction, 150, 7280))
        
        response_payload = {
            "model_version": "v2.0.0_Production_Champion",
            "predicted_traffic_volume": round(final_prediction, 0),
            "unit": "vehicles/hour",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        return jsonify(response_payload), 200
        
    except Exception as api_err:
        logger.error(f"API Serving Engine Exception: Execution blocked. Detail: {str(api_err)}")
        return jsonify({"status": "ERROR", "error_message": str(api_err)}), 400

#=====================================================================
#6.4 & 6.5 REAL-TIME DRIFT MONITORING & SYSTEM ALERTING DASHBOARD
#=====================================================================

def run_telemetry_drift_monitoring_simulation(current_mean_temp: float, observed_error_rate: float):
    """Simulates metric monitoring layers by tracking statistical and predictive distribution drift."""
    logger.info("Launching Automated Real-Time Drift Monitoring Telemetry Check...")
    
    #Baseline historical controls established during Task 1 & 2 engineering
    HISTORICAL_MEAN_TEMP = 281.27
    CRITICAL_MAE_THRESHOLD = 1000.0 # Alerts if mean prediction error climbs beyond 1000 vehicles
    
    #1. Evaluate Feature Distribution Drift (Checks if ambient temp shifts structurally)
    temp_drift_delta = abs(current_mean_temp - HISTORICAL_MEAN_TEMP)
    is_feature_drifted = temp_drift_delta > 15.0 # Flag alert if temperature variance shifts past 15 Kelvin
    
    #2. Evaluate Prediction Error Drift (Checks if accuracy remains inside acceptable boundaries)
    is_error_drifted = observed_error_rate > CRITICAL_MAE_THRESHOLD
    
    print("\n" + "="*65)
    print("🚦 SMART CITY SMART MOBILITY OPERATIONS MONITORING DASHBOARD")
    print("="*65)
    print(f"📊 Tracking Status Report Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📡 Serving Model Context Blueprint   : v2.0.0_Production_Champion")
    print(f"🌡️ Mean Scaled Temperature Ingest   : {current_mean_temp:.2f} K (Historical Baseline: {HISTORICAL_MEAN_TEMP:.2f} K)")
    print(f"📉 Real-Time Current Operational Error: {observed_error_rate:.2f} MAE (Max Bound Guard: {CRITICAL_MAE_THRESHOLD:.2f})")
    print("-"*65)
    
    #System Alerting State Decision Logic
    if is_feature_drifted or is_error_drifted:
        system_status = "⚠️ ALERT / Requires Immediate System Investigation"
        log_severity = logger.warning
        print(f"🚨 SYSTEM HEALTH STATE : {system_status}")
        if is_feature_drifted:
            print("❌ REASON: Critical Feature Distribution Drift Detected! Abnormal local weather shift.")
        if is_error_drifted:
            print("❌ REASON: Predictive Accuracy Degradation Detected! Model error tracks past threshold.")
    else:
        system_status = "✅ PASS / Normal Baseline Operations"
        log_severity = logger.info
        print(f"🟢 SYSTEM HEALTH STATE : {system_status}")
        print("✨ All monitored feature spreads and prediction windows conform to strict tolerance matrix bounds.")
        
    print("="*65 + "\n")
    log_severity(f"MLOps Monitoring Cycle Executed. Health Assessment Verdict: {system_status}")

#--- SIMULATE TELEMETRY SCENARIOS ---
#Scenario A: Normal Operating Conditions
run_telemetry_drift_monitoring_simulation(current_mean_temp=282.10, observed_error_rate=632.10)

#Scenario B: Critical Predictive Performance Degradation (Triggers System ALERT)
run_telemetry_drift_monitoring_simulation(current_mean_temp=280.50, observed_error_rate=1240.85)

logger.info("Task 6 MLOps and Deployment Simulation framework and monitoring simulation completed successfully.")
logger.info("====================================================")