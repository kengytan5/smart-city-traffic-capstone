import sys
import os
import logging
import pandas as pd
import numpy as np

#Configure the logging layout to print directly to the screen and features.log
#TIP: Change logging.INFO to logging.DEBUG to see the hidden calculations instantly!
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("pipeline.log", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Task 2: Feature Engineering")

input_file = "cleaned_traffic_data.csv"
output_file = "engineered_traffic_features.csv"

logger.info("Feature Engineering Workspace Initialized. Ready for sequential steps...")

logger.info("Starting Execution: Ingesting cleaned data")

try:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing source file: {input_file}. Make sure you ran Task 1: pipeline.py first!")
        
    df = pd.read_csv(input_file)
    
    # 1. Log the starting shape
    initial_rows, initial_cols = df.shape
    logger.info(f"Dataset shape BEFORE feature engineering: {initial_rows} rows, {initial_cols} columns.")

except FileNotFoundError as err:
    logger.error(f"Inbound Data Source Error: {str(err)}", exc_info=True)
    raise

logger.info("Engineering temporal features and cyclical math transformations")

#Force column type consistency
df['date_time'] = pd.to_datetime(df['date_time'])

#Extract base timeline categories
df['Hour'] = df['date_time'].dt.hour
df['Day_of_Week'] = df['date_time'].dt.dayofweek  # 0=Monday, 6=Sunday

#Create binary Weekend indicator (1 if Saturday or Sunday, else 0)
df['Is_Weekend'] = np.where(df['Day_of_Week'] >= 5, 1, 0)

#Apply Cyclical Encoding for Hour
max_hour_val = 24.0
logger.debug(f"Intermediate Variable: Cyclical Hour baseline boundary set to {max_hour_val}.")

df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / max_hour_val)
df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / max_hour_val)

logger.info("Time-based cyclical dimensions compiled successfully")

logger.info("Executing categorical weather element encoding arrays")

#One-hot encode high-level weather states (creates separate 0/1 binary indicator columns)
df_encoded = pd.get_dummies(df, columns=['weather_main'], prefix='Weather', dtype=int)

#Create appropriate derived indicator: Freezing Precipitation Hazard
#Logic: Flags 1 if temperature is below freezing (< 0°C / 273.15K) AND there is active rain or snow
df_encoded['Freezing_Precipitation_Hazard'] = np.where(
    (df_encoded['temp'] < 273.15) & ((df_encoded['rain_1h'] > 0) | (df_encoded['snow_1h'] > 0)), 
    1, 0
)

logger.info("Categorical one-hot matrices and environmental indices generated")

logger.info("Executing continuous feature normalization (Min-Max & Standardisation)")

#1. Z-score Standardisation for Temperature
temp_mean = df_encoded['temp'].mean()
temp_std = df_encoded['temp'].std()
logger.debug(f"Intermediate Metrics for 'temp' Scaling -> Mean: {temp_mean:.4f}, Std Dev: {temp_std:.4f}")

df_encoded['Scaled_Temp'] = (df_encoded['temp'] - temp_mean) / temp_std

#2. Min-Max Normalisation for Cloud Cover
cloud_min = df_encoded['clouds_all'].min()
cloud_max = df_encoded['clouds_all'].max()
logger.debug(f"Intermediate Metrics for 'clouds_all' Normalisation -> Min: {cloud_min}, Max: {cloud_max}")

df_encoded['Normalised_Clouds'] = (df_encoded['clouds_all'] - cloud_min) / (cloud_max - cloud_min)

logger.info("Continuous data ranges scaled for modeling compatibility")

logger.info("Calculating empirical traffic volume data distribution metrics")

#Calculate mathematical percentile cutoffs
q1_val = df_encoded['traffic_volume'].quantile(0.25)
q3_val = df_encoded['traffic_volume'].quantile(0.75)

logger.debug(f"Statistical Quartile Thresholds -> Lower 25th (Q1): {q1_val} vehicles, Upper 75th (Q3): {q3_val} vehicles.")

#Categorization Logic:
# - Low Flow: Volume is in the bottom 25% of all recorded hours (<= Q1)
# - Moderate Flow: Volume falls within normal interquartile daily bounds (between Q1 and Q3)
# - High Flow: Volume climbs into the heaviest 25% of commuter rush hours (>= Q3)
conditions = [
    (df_encoded['traffic_volume'] <= q1_val),
    (df_encoded['traffic_volume'] > q1_val) & (df_encoded['traffic_volume'] < q3_val),
    (df_encoded['traffic_volume'] >= q3_val)
]
categories = ['Low Flow', 'Moderate Flow', 'High Flow']

df_encoded['Congestion_Category'] = np.select(conditions, categories, default='Moderate Flow')

#Log final compiled dimensions
final_rows, final_cols = df_encoded.shape
logger.info(f"Dataset shape AFTER feature engineering: {final_rows} rows, {final_cols} columns.")

#Export final processed matrix
df_encoded.to_csv(output_file, index=False)
logger.info(f"Feature engineering completed successfully! Final ML-ready file saved as: {output_file}")

