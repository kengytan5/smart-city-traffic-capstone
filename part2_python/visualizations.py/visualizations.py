import sys
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Configure the logging layout to track figure exports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("pipeline.log", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Task 3: Visualize Traffic Patterns")
input_file = "engineered_traffic_features.csv"

logger.info("Visualisation Workspace Initialized. Ingesting features matrix...")

#Load the file created in Task 2
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Missing resource file: {input_file}. Make sure you ran Task 2: feature_engineering.py first!")

df = pd.read_csv(input_file)
df['date_time'] = pd.to_datetime(df['date_time'])
logger.info("Data ingested successfully. Analytical plotting loops ready")

def generate_visualizations(df: pd.DataFrame) -> None:
    """Compiles, customizes, and exports the six target analytical charts."""

logger.info("Generating Visualisation 1: Traffic demand by hour")

#Compute average hourly volume across the dataset
hourly_avg = df.groupby('Hour')['traffic_volume'].mean().reset_index()

#Construct the Matplotlib canvas
plt.figure(figsize=(10, 5))
plt.plot(hourly_avg['Hour'], hourly_avg['traffic_volume'], color='#2c3e50', linewidth=2.5, marker='o')

#Apply professional grid styling and text overlays
plt.title("Traffic Demand By Hour", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Hour of the Day (00:00 - 23:00)", fontsize=11, labelpad=10)
plt.ylabel("Average Traffic Volume (Vehicles/Hour)", fontsize=11, labelpad=10)
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

#Save figure to hard drive and execute mandatory compliance log
fig1_path = "traffic_demand_by_hour.png"
plt.savefig(fig1_path, dpi=300)
plt.close()

logger.info(f"Successfully saved figure 1: {fig1_path}")

logger.info("Generating Visualisation 2: Weekday vs. weekend traffic")

#Construct a readable string flag to handle structural categorization plotting labels
df['Day_Type'] = np.where(df['Is_Weekend'] == 1, 'Weekend', 'Weekday')

#Group hourly averages across separate day classes
day_type_avg = df.groupby(['Hour', 'Day_Type'])['traffic_volume'].mean().unstack()

#Construct the Matplotlib dual column canvas
fig, ax = plt.subplots(figsize=(12, 6))
x_axis_indices = np.arange(24)
bar_width = 0.4

ax.bar(x_axis_indices - bar_width/2, day_type_avg['Weekday'], bar_width, label='Weekdays (Mon-Fri)', color='#e74c3c')
ax.bar(x_axis_indices + bar_width/2, day_type_avg['Weekend'], bar_width, label='Weekends (Sat-Sun)', color='#3498db')

#Apply visual configurations
ax.set_title("Weekday vs. Weekend Traffic by Hour", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Hour of the Day", fontsize=11, labelpad=10)
ax.set_ylabel("Average Traffic Volume", fontsize=11, labelpad=10)
ax.set_xticks(x_axis_indices)
ax.legend(frameon=True, facecolor='white', edgecolor='none')
ax.grid(True, linestyle='--', alpha=0.4, axis='y')
plt.tight_layout()

#Export figure and record log trace path
fig2_path = "weekday_vs_weekend_traffic.png"
plt.savefig(fig2_path, dpi=300)
plt.close()

logger.info(f"Successfully saved figure 2: {fig2_path}")

logger.info("Generating Visualisation 3: Traffic and weather relationships")

#1. Identify all weather one-hot columns (e.g., Weather_Clear, Weather_Rain)
weather_cols = [col for col in df.columns if col.startswith('Weather_')]

#2. Calculate the mean traffic volume for each column when it is active (equals 1)
#This completely bypasses the KeyError since it reads the hot-encoded fields directly
weather_means = {col.replace('Weather_', ''): df[df[col] == 1]['traffic_volume'].mean() for col in weather_cols}

#3. Sort the results ascendingly to match your horizontal plot layout requirements
sorted_weather = sorted(weather_means.items(), key=lambda x: x[1])

#4. Extract categories and numeric arrays for Matplotlib mapping
categories = [item[0] for item in sorted_weather]
mean_values = [item[1] for item in sorted_weather]

#5. Construct horizontal bar plot canvas layout
plt.figure(figsize=(10, 6))
colors_list = plt.cm.viridis(np.linspace(0.3, 0.7, len(categories)))
plt.barh(categories, mean_values, color=colors_list, height=0.6)

#Apply labeling and formatting anchors
plt.title("Traffic and Weather Relationships", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Mean Hourly Traffic Volume (Vehicles/Hour)", fontsize=11, labelpad=10)
plt.ylabel("Weather Category", fontsize=11, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.4, axis='x')
plt.tight_layout()

#Export final figure artifact and finalize validation log trail
fig3_path = "traffic_weather_relationships.png"
plt.savefig(fig3_path, dpi=300)
plt.close()

#6. Compliance Trail Logs mapping out the calculated distributions safely
logger.info(f"Successfully saved figure 3: {fig3_path}")
for weather, mean_val in sorted_weather:
    logger.info(f"Calculated Baseline Metric - {weather}: {mean_val:.2f} vehicles/hour")

logger.info("Generating Visualisation 4: Temperature vs. traffic")
    
#Exclude unphysical 0 Kelvin sensor defects to keep the visual grid strictly focused
legitimate_weather = df[df['temp'] > 50].copy()
    
#Safely standardise Kelvin logs back into readable Celsius format if required
if legitimate_weather['temp'].mean() > 200:
    legitimate_weather['temp_celsius'] = legitimate_weather['temp'] - 273.15
else:
    legitimate_weather['temp_celsius'] = legitimate_weather['temp']
        
plt.figure(figsize=(10, 6))
plt.scatter(legitimate_weather['temp_celsius'], legitimate_weather['traffic_volume'], alpha=0.15, color='#2980b9', s=4)
    
plt.title("Temperature vs. Traffic", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Temperature (°C)", fontsize=11, labelpad=10)
plt.ylabel("Hourly Traffic Volume (Vehicles/Hour)", fontsize=11, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
    
fig4_path = "temperature_vs_traffic.png"
plt.savefig(fig4_path, dpi=300)
plt.close()
logger.info(f"Successfully saved figure 4: {fig4_path}")

logger.info("Generating Visualisation 5: Traffic Distribution")
    
plt.figure(figsize=(10, 6))
plt.hist(df['traffic_volume'], bins=30, color='#8e44ad', edgecolor='white', alpha=0.8)
    
plt.title("Traffic Distribution", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Hourly Traffic Volume (Vehicles/Hour)", fontsize=11, labelpad=10)
plt.ylabel("Frequency Count (Total Logged Hours)", fontsize=11, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.tight_layout()
    
fig5_path = "traffic_distribution.png"
plt.savefig(fig5_path, dpi=300)
plt.close()
logger.info(f"Successfully saved figure 5: {fig5_path}")

logger.info("Generating Visualisation 6: Congestion pattterns over time")
    
#Re-extract structural timeline dimensions if missing
df['Year'] = df['date_time'].dt.year
df['Month_Idx'] = df['date_time'].dt.month
    
#Isolate exclusive high flow congestion rows
high_flow_history = df[df['Congestion_Category'] == 'High Flow']
    
#Pivot rows into a grid structured by Year and Month indices
timeline_counts = high_flow_history.groupby(['Year', 'Month_Idx']).size().unstack(level=0)
months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
plt.figure(figsize=(12, 6))
    
#Programmatically graph tracking lines for each year available in the matrix
if 2015 in timeline_counts.columns:
    plt.plot(months_labels, timeline_counts[2015], marker='s', linewidth=2, label='2015 Operations', color='#c0392b')
if 2016 in timeline_counts.columns:
    plt.plot(months_labels, timeline_counts[2016], marker='o', linewidth=2, label='2016 Operations', color='#27ae60')
if 2017 in timeline_counts.columns:
    plt.plot(months_labels, timeline_counts[2017], marker='^', linewidth=2, label='2017 Operations', color='#f39c12')
        
plt.title("Congestions Patterns Over Time", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Calendar Month", fontsize=11, labelpad=10)
plt.ylabel("Number of Logged Congestion Hours (>5,500 Vehicles)", fontsize=11, labelpad=10)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
    
fig6_path = "congestion_patterns_over_time.png"
plt.savefig(fig6_path, dpi=300)
plt.close()
logger.info(f"Successfully saved figure 6: {fig6_path}")

logger.info("Execution completed successfully. All image files compiled")

def main():
    """Execution Handler Ingestion Loop."""
    logger.info("====================================================")
    logger.info("LAUNCHING AUTOMATED VISUALIZATION COMPILATION PHASE")
    logger.info("====================================================")
    
    if not os.path.exists(input_file):
        logger.error(f"Inbound Data Error: {input_file} missing. Run Task 2: feature_engineering.py first.")
        sys.exit(1)
        
    df = pd.read_csv(input_file)
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    generate_visualizations(df)
    
    logger.info("Visualization engine executed successfully. Output artifacts ready.")
    logger.info("====================================================")


if __name__ == "__main__":
    main()


