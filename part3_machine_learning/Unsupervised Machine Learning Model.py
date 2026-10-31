import sys
import os
import logging
import pandas as pd
import numpy as np

#Machine Learning & Clustering Toolkits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Association Rule Mining Toolkits
from mlxtend.frequent_patterns import apriori, association_rules

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

logger = logging.getLogger("Unsupervised Machine Learning")
data_source = "ml_traffic_accident_proxy.csv"

logger.info("====================================================")
logger.info("LAUNCHING TASK 2: UNSUPERVISED MACHINE LEARNING WORKSPACE")
logger.info("====================================================")

if not os.path.exists(data_source):
    logger.error(f"Inbound Data Source Error: {data_source} missing. Execute Proxy accident-risk label.py first.")
    sys.exit(1)

df = pd.read_csv(data_source)
logger.info(f"Successfully loaded dataset cache. Shape: {df.shape}")

logger.info("Initializing Unsupervised Stage: K-Means Clustering on Traffic Conditions")

#1. Map an environmental severity index metric to combine with hour and volume features
weather_cols = [col for col in df.columns if col.startswith('Weather_') and col != 'Weather_Unknown']
df['Weather_Severity_Index'] = df[weather_cols].multiply(np.arange(1, len(weather_cols) + 1), axis=1).sum(axis=1)

#Isolate features for clustering
cluster_features = ['Hour', 'traffic_volume', 'Weather_Severity_Index']
X_cluster = df[cluster_features]

#Standardize the continuous feature space to establish equal scaling weight
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

#Run K-Means with an optimized 3-cluster initialization
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
df['Traffic_Cluster'] = kmeans.fit_transform(X_cluster_scaled).argmin(axis=1) # Fixed syntax to cleanly assign labels

#Extract the centroids/mean profiles of each generated cluster for interpretation
cluster_profiles = df.groupby('Traffic_Cluster')[cluster_features].mean()

for cluster_id in range(3):
    profile = cluster_profiles.loc[cluster_id]
    logger.info(
        f"Cluster {cluster_id} Profile -> Mean Hour: {profile['Hour']:.1f} | "
        f"Mean Volume: {profile['traffic_volume']:.0f} vehicles/hr | "
        f"Weather Severity Index: {profile['Weather_Severity_Index']:.2f}"
    )

logger.info("Initializing Unsupervised Stage: Association Rule Mining (Apriori)")

#1. Discretize timeline and volume dimensions into explicit text buckets
df['Time_of_Day'] = pd.cut(df['Hour'], bins=[-1, 6, 9, 15, 18, 24], 
                           labels=['Night', 'Morning_Peak', 'Midday', 'Evening_Peak', 'Night_Late'])
df['Day_Type'] = np.where(df['Is_Weekend'] == 1, 'Weekend', 'Weekday')

#Use your engineered 4-tier target for the congestion level tracking parameter
df['Congestion_Level'] = df['congestion_category']

#Isolate categorical fields for market-basket transaction transformation
basket_source = df[['Time_of_Day', 'Day_Type', 'weather_main', 'Congestion_Level']]

#Convert categories into a wide binary one-hot encoded boolean transaction matrix
basket_encoded = pd.get_dummies(basket_source, dtype=bool)

#2. Mine frequent itemsets with a baseline minimum support threshold of 2%
frequent_itemsets = apriori(basket_encoded, min_support=0.02, use_colnames=True)

#3. Generate association rules evaluated via a confidence cutoff of 40%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)

#Sort rules to isolate patterns with the highest lift scores
highest_lift_rules = rules.sort_values(by="lift", ascending=False).head(3)

print("\n" + "="*70)
print("🚨 TOP MINED ASSOCIATION RULES WITH HIGHEST LIFT METRICS")
print("="*70)

for idx, row in highest_lift_rules.iterrows():
    antecedents = list(row['antecedents'])
    consequents = list(row['consequents'])
    
    print(f"Rule: If {antecedents} -> Then {consequents}")
    print(f" 📊 Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.4f}")
    
    # Structural logging check
    logger.info(f"Mined Rule Pattern found: {antecedents} -> {consequents} (Lift: {row['lift']:.4f})")

print("="*70)
logger.info("Task 2: Unsupervised Machine Learning Pipeline execution successfully finalized.")
logger.info("====================================================")