import sys
import os
import logging
import pandas as pd
import numpy as np

#Configure logging to append to your unified log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("pipeline.log", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Part 3: Proxy accident-risk label")
input_file = "engineered_traffic_features.csv"
output_file = "ml_traffic_accident_proxy.csv"

logger.info("====================================================")
logger.info("STARTING PART 3: PROXY ACCIDENT-RISK LABEL")
logger.info("====================================================")

#1. Ingest Feature Matrix from Part 2
if not os.path.exists(input_file):
    logger.error(f"Inbound Data Error: {input_file} missing. Make sure you ran Task 2: feature_engineering.py first!")
    sys.exit(1)

df = pd.read_csv(input_file)
logger.info(f"Successfully loaded feature data. Starting shape: {df.shape}")

#2. Reconstruct original text categories if dropped by one-hot encoding
#This ensures we can scan 'weather_main' as requested by the prompt
if 'weather_main' not in df.columns:
    weather_cols = [col for col in df.columns if col.startswith('Weather_')]
    df['weather_main'] = df[weather_cols].idxmax(axis=1).str.replace('Weather_', '')
    logger.info("Reconstructed 'weather_main' categorical variable from one-hot matrices.")

#=====================================================================
#CREATE CONGESTION CATEGORIES (4-TIER QUARTILES)
#=====================================================================
#Extract empirical quartile boundaries from the continuous traffic volume
q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.50, 0.75]).values
logger.info(f"Extracted Traffic Quartiles -> Q1: {q1:.0f}, Median(Q2): {q2:.0f}, Q3: {q3:.0f}")

def bucket(v):
    if v <= q1:
        return "Low"
    elif v <= q2:
        return "Medium"
    elif v <= q3:
        return "High"
    return "Severe"

df["congestion_category"] = df["traffic_volume"].apply(bucket)
logger.info("Successfully generated 4-tier 'congestion_category' targets")

#=====================================================================
#DEFINE ENVIRONMENTAL HAZARD BOUNDARIES & PROXY ACCIDENT-RISK
#=====================================================================
#Define which weather states represent physical hazards based on Part 2 findings
SEVERE_WEATHER = ['Squall', 'Thunderstorm', 'Snow', 'Rain', 'Drizzle']

#Flag low visibility hazards if any restricting atmospheric conditions are active
df['is_low_visibility'] = np.where(df['weather_main'].isin(['Fog', 'Haze', 'Mist', 'Smoke']), 1, 0)

#Evaluate logical masks
high_congestion = df["congestion_category"].isin(["High", "Severe"])

risky_weather = (
    df["weather_main"].isin(SEVERE_WEATHER)
    | (df["is_low_visibility"] == 1)
)

#Compile proxy accident-risk target vector (1 for high risk, 0 for normal)
df["high_risk"] = (high_congestion & risky_weather).astype(int)

# Count metrics for verification
total_high_risk = df["high_risk"].sum()
percentage_risk = (total_high_risk / len(df)) * 100

logger.warning(
    f"Proxy Generation Warning: Synthesized {total_high_risk} high-risk rows "
    f"({percentage_risk:.2f}% of total hours) based on severe conditions constraint logic."
)

#Save final machine learning modeling dataset
df.to_csv(output_file, index=False)
logger.info(f"Target dataset created successfully. Saved artifact as: {output_file}")
logger.info("====================================================")

