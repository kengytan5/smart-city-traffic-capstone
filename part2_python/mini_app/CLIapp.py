import sys
import os
import logging
import pandas as pd
import numpy as np

#Configure the logging framework to trace user interactive commands
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("pipeline.log", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Task 4: Build a Mini Traffic Analytics Application")
data_source = "engineered_traffic_features.csv"

logger.info("Initializing Smart City CLI Application Workspace...")

#Verify database artifact availability
if not os.path.exists(data_source):
    logger.error(f"Inbound data missing: {data_source} not found. Run Task 2 first!")
    raise FileNotFoundError(f"Missing engineered dataset asset.")

df = pd.read_csv(data_source)
df['date_time'] = pd.to_datetime(df['date_time'])
logger.info(f"Database fully loaded into CLI cache matrix. Row Count: {df.shape[0]}")

def query_specific_datetime(target_str):
    """Command 1: Lookup weather, category, and density metrics for an exact hour."""
    logger.info(f"Command Invoked: datetime-lookup | Arguments: target_date='{target_str}'")
    try:
        # Convert user string into true datetime object for query mapping
        parsed_target = pd.to_datetime(target_str)
        
        # Filter matching database records exactly
        record = df[df['date_time'] == parsed_target]
        
        if record.empty:
            print(f"\n[⚠️] No recorded sensor log found matching: {parsed_target}")
            logger.info("Command Complete: Execution successful (Zero results returned)")
            return
        
        # Pull one-hot category tracking safely without KeyError traps
        weather_cols = [col for col in record.columns if col.startswith('Weather_')]
        active_weather = "Unknown"
        for col in weather_cols:
            if record[col].values[0] == 1:
                active_weather = col.replace('Weather_', '')
                break
                
        print("\n" + "="*50)
        print(f"📊 TRAFFIC METRICS FOR: {parsed_target}")
        print("="*50)
        print(f"🚘 Vehicle Volume    : {record['traffic_volume'].values[0]} vehicles/hour")
        print(f"🚦 Congestion Class  : {record['Congestion_Category'].values[0]}")
        print(f"🌤️ Weather Baseline  : {active_weather}")
        print(f"🌡️ Temperature (Raw) : {record['temp'].values[0]:.2f} Kelvin")
        print(f"🥶 Freezing Hazard   : {'YES (⚠️ High Risk)' if record['Freezing_Precipitation_Hazard'].values[0] == 1 else 'No'}")
        print("="*50)
        logger.info("Command Complete: Operational database lookup finished successfully")
        
    except Exception:
        # Fulfills mandate to report clean ERROR logs instead of raw traceback strings
        logger.error(f"Invalid Entry: The string '{target_str}' is malformed or invalid. Use YYYY-MM-DD HH:MM:00 layout.")
        print("\n[❌ Error] Datetime lookup failed. Please input a valid format (Example: 2017-05-15 08:00:00).")


def identify_high_traffic_periods(threshold=5500):
    """Command 2: Isolate hours where interstate volume hits maximum carrying constraints."""
    logger.info(f"Command Invoked: congestion-lookup | Arguments: threshold={threshold}")
    
    if threshold < 0 or threshold > 8000:
        logger.error(f"Invalid Entry: Out-of-bounds capacity metric supplied: {threshold}")
        print(f"\n[❌ Error] Plausible highway capacity ranges strictly between 0 and 8000 vehicles.")
        return
        
    high_flow_records = df[df['traffic_volume'] > threshold]
    total_congested_hours = high_flow_records.shape[0]
    
    print("\n" + "="*50)
    print(f"🚨 HIGH-TRAFFIC SATURATION REPORT (> {threshold} vehicles)")
    print("="*50)
    print(f"📉 Total Gridlock Hours Found : {total_congested_hours} out of {df.shape[0]}")
    
    if total_congested_hours > 0:
        # Aggregate patterns by day type and hour metrics
        peak_hours = high_flow_records['Hour'].value_counts().head(3).index.tolist()
        print(f"⏰ Top 3 Most Congested Hours : {', '.join([f'{h:02d}:00' for h in peak_hours])}")
        
        weekend_shares = high_flow_records['Is_Weekend'].mean() * 100
        print(f"📅 Weekend Share Percentage  : {weekend_shares:.2f}% of high traffic events")
    print("="*50)
    logger.info("Command Complete: Congestion analysis mapping loop successfully logged")

def compare_weekday_weekend_traffic():
    """Command 4: Compute and output comparative metrics between Weekdays and Weekends."""
    logger.info("Command Invoked: weekday-weekend-comparison | Arguments: None")
    
    # 1. Isolate Weekday vs Weekend splits using our pre-engineered binary flag
    weekday_data = df[df['Is_Weekend'] == 0]
    weekend_data = df[df['Is_Weekend'] == 1]
    
    # 2. Compute key continuous data metrics
    avg_weekday = weekday_data['traffic_volume'].mean()
    avg_weekend = weekend_data['traffic_volume'].mean()
    
    max_weekday = weekday_data['traffic_volume'].max()
    max_weekend = weekend_data['traffic_volume'].max()
    
    # 3. Calculate high congestion probability thresholds (>5500 vehicles)
    weekday_cong_prob = (weekday_data['traffic_volume'] > 5500).mean() * 100
    weekend_cong_prob = (weekend_data['traffic_volume'] > 5500).mean() * 100
    
    print("\n" + "="*50)
    print("📊 INTERSTATE FLOW COMPARISON: WEEKDAYS VS WEEKENDS")
    print("="*50)
    print(f"📈 Average Volume (Hourly) : Weekdays: {avg_weekday:.1f} | Weekends: {avg_weekend:.1f}")
    print(f"💥 Absolute Peak Volume    : Weekdays: {max_weekday:.0f}  | Weekends: {max_weekend:.0f}")
    print(f"🚨 Risk of Congestion (>5.5K): Weekdays: {weekday_cong_prob:.2f}% | Weekends: {weekend_cong_prob:.2f}%")
    print("-"*50)
    print(f"💡 Insight: Weekday hourly averages are ~{avg_weekday - avg_weekend:.0f} vehicles higher")
    print(f"   due to structured workforce commute patterns.")
    print("="*50)
    
    logger.info("Command Complete: Weekday vs Weekend analytical matrix generated successfully")

def recommend_travel_periods():
    """Command 3: Calculate data-driven optimized commute scheduling frameworks."""
    logger.info("Command Invoked: optimize-commute | Arguments: None")
    
    # Calculate baseline metrics isolating standard daylight operational loops (6 AM to 8 PM)
    daylight_df = df[(df['Hour'] >= 6) & (df['Hour'] <= 20)]
    
    # Group by operational hour tracking indices to isolate minimum averages
    hourly_means = daylight_df.groupby('Hour')['traffic_volume'].mean()
    best_daylight_hours = hourly_means.sort_values(ascending=True).head(3).index.tolist()
    
    print("\n" + "="*50)
    print("💡 OPTIMIZED INTERSTATE TRAVEL RECOMMENDATIONS")
    print("="*50)
    print("Top 3 Safest Daylight Windows (Lowest Traffic Volumes):")
    for rank, hr in enumerate(best_daylight_hours, start=1):
        print(f" {rank}. {hr:02d}:00 (Mean Flow: {hourly_means[hr]:.0f} vehicles/hour)")
    
    print("\n⚠️ Travel Warning Patterns:")
    print(" • Avoid 07:00 - 08:00 and 16:00 - 17:00 due to severe bimodal rush-hour surges.")
    print(" • Double-check your dashboard filters when Freezing Hazard equals 1.")
    print("="*50)
    logger.info("Command Complete: Travel window recommendations generated successfully.")

def retrieve_weather_traffic_info(weather_query):
    """Command 5: Filter the master cache to find traffic volume metrics for a specific weather type."""
    # Clean the input token string to match standard database storage layouts
    clean_query = weather_query.strip().capitalize()
    logger.info(f"Command Invoked: weather-traffic-lookup | Arguments: query_term='{clean_query}'")
    
    # Construct the one-hot variable string we generated in Task 2
    target_column = f"Weather_{clean_query}"
    
    # Look for the column inside the active data structure
    if target_column not in df.columns:
        logger.error(f"Invalid Entry: Weather category '{clean_query}' does not exist in dataset schema.")
        print(f"\n[❌ Error] Weather category '{clean_query}' not recognized.")
        
        # Pull a list of valid active features to show the user
        valid_options = [col.replace('Weather_', '') for col in df.columns if col.startswith('Weather_')]
        print(f"💡 Valid weather options are: {', '.join(valid_options)}")
        return

    # Filter where the target one-hot condition is true (equals 1)
    weather_subset = df[df[target_column] == 1]
    
    # Compute aggregates
    mean_flow = weather_subset['traffic_volume'].mean()
    max_flow = weather_subset['traffic_volume'].max()
    sample_size = len(weather_subset)
    
    print("\n" + "="*50)
    print(f"🌤️ CLIMATE INTELLIGENCE REPORT: {clean_query.upper()}")
    print("="*50)
    print(f"📊 Historical Hours Logged : {sample_size} hours")
    print(f"🚘 Mean Hourly Traffic Flow: {mean_flow:.1f} vehicles/hour")
    print(f"🚀 Peak Traffic Capacity   : {max_flow:.0f} vehicles/hour")
    print("="*50)
    
    logger.info(f"Command Complete: Weather conditions extraction complete for category '{clean_query}'.")

# =====================================================================
# INTERACTIVE APPLICATION PROMPT CONTROL LOOP
# =====================================================================
print("\n" + "*"*60)
print("🚀 WELCOME TO THE SMART CITY TRAFFIC INTELLIGENCE ENGINE")
print("*"*60)

while True:
    print("\n[🤖] Main Command Menu:")
    print("  1. Query Traffic for a specific date/time")
    print("  2. Identify high-traffic periods")
    print("  3. Compare weekday and weekend traffic")
    print("  4. Identify recommended travel periods")
    print("  5. Retrieve weather and traffic information")
    print("  6. Terminate Engine (Exit Application)")
    
    user_choice = input("\nSelect a command option (1-4): ").strip()
    
    if user_choice == '1':
        target_date = input("Enter timestamp (Format: YYYY-MM-DD HH:MM:00, e.g., 2017-05-15 17:00:00): ").strip()
        query_specific_datetime(target_date)
        
    elif user_choice == '2':
        threshold_input = input("Enter traffic capacity limit metric (Default: 5500): ").strip()
        if threshold_input == "":
            identify_high_traffic_periods()
        else:
            try:
                identify_high_traffic_periods(threshold=int(threshold_input))
            except ValueError:
                logger.error(f"Invalid Entry: Supplied threshold string '{threshold_input}' cannot be cast to Integer.")
                print("\n[❌ Error] Capacity limits must be entered as whole number integers.")
                
    elif user_choice == '3':
        compare_weekday_weekend_traffic()
        
    elif user_choice == '4':
        recommend_travel_periods()

    elif user_choice == '5':
        weather_type = input("Enter weather type (e.g., Rain, Clear, Snow, Clouds): ").strip()
        retrieve_weather_traffic_info(weather_type)
        
    elif user_choice == '6':
        logger.info("User requested exit sequence. Shutting down CLI interactive application engine")
        print("\nClosing workspace database cache connection. Goodbye!")
        break
        
    else:
        logger.error(f"Invalid Entry: User entered invalid command index string: '{user_choice}'")
        print("\n[⚠️ Warning] Selection out of bounds. Please input a number option between 1 and 6.")