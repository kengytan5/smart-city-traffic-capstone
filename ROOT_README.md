# smart-city-traffic-capstone

AI, ML and Data Science Capstone Project





**Root\_Readme.md overall file for all 3 parts**



This end-to-end data and AI capstone project supports a city government's Smart City Mobility Analytics Team in improving traffic management and commuter mobility. Using approximately 48,000 hourly traffic, weather, and holiday records for westbound I-94, the project progresses from exploratory data analytics and business intelligence to reproducible data engineering and machine learning/AI deployment.





**Project Structure \& Component Breakdown**



smart-city-traffic-capstone/

│

├── part1\_data\_analytics/                #SQL analysis, statistical \& probability modeling, and Power BI dashboard

│

├── part2\_python/                        #Reproducible Python data pipeline, feature engineering, visualizations \& CLI app

│

├── part3\_machine\_learning/              #Supervised/unsupervised/deep learning models, explainability, MLOps \& Responsible AI

│

├── .gitattributes

│

└── ROOT\_README.md                       #Root Overview of the project information







**Part 1: Data Analytics**

&#x09;1. SQL-Based Data Analysis: Querying hourly, daily, seasonal, and weather-driven traffic volume trends.

&#x09;2. Descriptive Statistics \& Correlation: Evaluating summary metrics, distributions, and linear/non-linear relationships between weather variables and traffic flow.

&#x09;3. Probability \& Conditional Probability: Calculating congestion probabilities, conditional probabilities given severe weather or peak hours, independence tests, and odds ratios.

&#x09;4. Power BI Dashboard: Interactive visual report synthesising traffic patterns, weather impacts, and actionable mobility insights.



**Part 2: Python** 

&#x09;1. Reproducible Data Pipeline: Automated ingestion, schema validation, missing-value handling, and outlier treatment with structured logging.

&#x09;2. Feature Engineering: Cyclical time encoding, peak-hour indicators, lag/rolling traffic features, and weather severity indices.

&#x09;3. Visualisations: Exploratory and diagnostic plots saved for reporting.

&#x09;4. Command-Line Application (CLI): Interactive terminal tool allowing users to query cleaned traffic summaries, run pipeline stages, and inspect mobility metrics.



**Part 3: Machine Learning**

&#x09;1. Predictive \& Pattern Modelling: Supervised regression/classification models, unsupervised clustering of traffic regimes, and deep learning time-series/tabular architectures.

&#x09;2. Model Explainability: Feature importance and SHAP/interpretation analysis to explain congestion drivers to city planners.

&#x09;3. Recommendation \& MLOps: Actionable traffic management recommendations, model serialization, performance tracking, and drift monitoring considerations.

&#x09;4. Responsible AI \& Governance: Critical evaluation of dataset limitations, geographic/demographic bias, fairness, environmental sustainability, and real-world deployment risks.







**Key Takeaways \& Impact**

&#x09;1. Peak Congestion Drivers: Time-of-day commuter cycles (morning and evening rush hours) and day-of-week patterns are the primary drivers of westbound I-94 volume, modulated by extreme temperatures and precipitation.

&#x09;2. End-to-End Reproducibility: Modular pipeline scripts, structured logs, and incremental version control ensure all analytical and modelling outputs are reproducible.

&#x09;3. Decision-Ready Intelligence: Combines interactive BI dashboards, CLI tooling, and explainable ML predictions to support proactive traffic signal timing, transit scheduling, and commuter alerts.



