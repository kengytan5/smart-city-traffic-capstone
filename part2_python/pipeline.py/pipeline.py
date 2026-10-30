import sys
import os
import logging
import pandas as pd
import numpy as np

#Configure the logging layout to print directly to the screen and pipeline.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("pipeline.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

#Instantiate our custom module logger
logger = logging.getLogger("Task 1: Data Pipeline Construction")

#Define the file configuration baseline
csv_filename = r'C:\Users\kw\OneDrive\NUSSOC\Week 33 - Capstone Project - Smart City Traffic Intelligence - From Data Analytics to AI-Powered Mobility\Metro_Interstate_Traffic_Volume.csv'
EXPECTED_COLUMNS = [
    'holiday', 'temp', 'rain_1h', 'snow_1h', 
    'clouds_all', 'weather_main', 'weather_description', 
    'date_time', 'traffic_volume'
]

logger.info("Pipeline Engine Ready. Beginning sequential execution...")

logger.info("Starting Execution: Data Loading")

try:
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"Target CSV file not found at: {csv_filename}")
        
#Read the dataset into our working DataFrame
    df = pd.read_csv(csv_filename)
    
    rows, cols = df.shape
    logger.info(f"Successfully loaded raw data file. Dimensions: {rows} rows, {cols} columns.")

except FileNotFoundError as fnf_err:
    logger.error(f"File System Error: {str(fnf_err)}", exc_info=True)
    raise
except pd.errors.ParserError as parse_err:
    logger.error(f"CSV Parse Failure: Check structure compatibility. Details: {str(parse_err)}", exc_info=True)
    raise

logger.info("Starting Execution: Schema Validation Validation")
current_columns = list(df.columns)

#Look for missing layout features
missing_columns = [col for col in EXPECTED_COLUMNS if col not in current_columns]

if missing_columns:
    error_msg = f"Schema structural verification failed! Missing fields: {missing_columns}"
    logger.error(error_msg)
    raise ValueError(error_msg)
else:
    logger.info("Schema structural verification successful. All mandatory target features are present.")

#1. Parse and Validate Date/Time
logger.info("Executing Cleaning Step: Parsing timestamps")
df['date_time'] = pd.to_datetime(df['date_time'])
df['Month'] = df['date_time'].dt.month
logger.info("Datetime fields converted successfully. Operational 'Month' index generated.")

#2. Identify and Remove Duplicate Rows
logger.info("Executing Cleaning Step: Deduplication")
duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    logger.warning(f"Deduplication Step: Dropped {duplicate_count} exact duplicate rows from the dataset.")
else:
    logger.info("Deduplication Step complete: No duplicate records detected")

#1. Standardise Categorical Values
logger.info("Executing Cleaning Step: Categorical Standardisation")
for cat_col in ['holiday', 'weather_main', 'weather_description']:
    if cat_col in df.columns:
        df[cat_col] = df[cat_col].astype(str).str.strip()
logger.info("Categorical string tracking alignment completed")

#2. Handle Temperature Outliers (0 Kelvin checks)
logger.info("Executing Cleaning Step: Temperature Outlier Assessment")
temp_outliers_mask = df['temp'] <= 0
temp_affected_rows = temp_outliers_mask.sum()

if temp_affected_rows > 0:
    # Explicit loop context: Process month-by-month to apply realistic local seasonal medians
    unique_months = df['Month'].unique()
    for month in unique_months:
        valid_monthly_temp = df[(df['Month'] == month) & (df['temp'] > 0)]['temp']
        monthly_median = valid_monthly_temp.median() if not valid_monthly_temp.empty else 278.15

        # Apply the seasonal baseline change to matching rows
        df.loc[(df['Month'] == month) & (temp_outliers_mask), 'temp'] = monthly_median
        
    logger.warning(
        f"Outlier Imputation: Modified {temp_affected_rows} rows exhibiting 0 Kelvin anomalies. "
        f"Substituted values with localized monthly medians."
    )
else:
    logger.info("Temperature record assessment complete: No 0 Kelvin errors observed")

#1. Handle Rainfall Outliers (> 500mm/hr is physically impossible)
logger.info("Executing Cleaning Step: Rainfall Outlier Assessment")
rain_outliers_mask = df['rain_1h'] > 500
rain_affected_rows = rain_outliers_mask.sum()

if rain_affected_rows > 0:
    # Explicit loop context: Process month-by-month for localized rainfall mapping
    for month in df['Month'].unique():
        valid_monthly_rain = df[(df['Month'] == month) & (df['rain_1h'] <= 500)]['rain_1h']
        monthly_median_rain = valid_monthly_rain.median() if not valid_monthly_rain.empty else 0.0
        
        df.loc[(df['Month'] == month) & (rain_outliers_mask), 'rain_1h'] = monthly_median_rain
        
    logger.warning(
        f"Outlier Imputation: Modified {rain_affected_rows} rows exhibiting physically implausible "
        f"precipitation totals. Values replaced using localized monthly median rainfall matrices."
    )
else:
    logger.info("Rainfall tracking audit complete: No critical value failures observed")

#2. Save Data Artifact
output_filename = "cleaned_traffic_data.csv"
df.to_csv(output_filename, index=False)

logger.info(f"Interactive Pipeline execution finished successfully. Cleaned matrix saved to: {output_filename}")



