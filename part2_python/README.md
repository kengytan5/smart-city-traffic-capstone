# smart-city-traffic-capstone

AI, ML and Data Science Capstone Project





**Instructions for Part 2 - Python: Building a Reproducible Traffic Analytics Pipeline**





**Project Structure \& Component Breakdown**



part2\_python/

├── cli\_app                                                              #Task 4: Interactive Command-Line Query Application Engine

├── feature\_engineering.py                                               #Task 2: Advanced Feature Generation \& Range Scaling Module

├── figures

│   ├── congestion\_patterns\_over\_time.png                                #Task 3 Visualisation 6

│   ├── temperature\_vs\_traffic.png                                       #Task 3 Visualisation 4

│   ├── traffic\_demand\_by\_hour.png                                       #Task 3 Visualisation 1

│   ├── traffic\_distribution.png                                         #Task 3 Visualisation 5

│   ├── traffic\_weather\_relationships.png                                #Task 3 Visualisation 3

│   └── weekday\_vs\_weekend\_traffic.png                                   #Task 3 Visualisation 2

├── notebooks/

│   ├── Assignment 33 Part 2 - Task 1.ipynb                              #Task 1

│   ├── Assignment 33 Part 2 - Task 2.ipynb                              #Task 2

│   ├── Assignment 33 Part 2 - Task 3.ipynb                              #Task 3

│   └── Assignment 33 Part 2 - Task 4.ipynb                              #Task 4

├── pipeline.log                                                         #Unified chronological trail file of all pipeline events

├── pipeline.py                                                          #Task 1: Ingestion, Validation, Cleaning \& Imputation Module

├── visualizations.py                                                    #Task 3: Six-Chart Automated Plotting Module

├── gitkeep.txt                                                          #This is a community-driven convention used to force Git to track an otherwise empty directory                                                                                                             ├── cleaned\_traffic\_data.csv                                             #Intermediate processed dataset (Output of Task 1)

├── engineered\_traffic\_features.csv                                      #Final ML-ready dataset (Output of Task 2), ready for Task 3 and 4

├── Metro\_Interstate\_Traffic\_Volume.csv                                  #Source file for Task 1

├── Part 2 - Executive Summary of Methodology and Findings Report.pdf    #1–2 summarising the methodology and findings

└── README.md                                                            #Comprehensive replication guide and system layout guide





**Step by Step Execution Sequence**



1. Before executing any .py file, under pipeline.py, in line 2, please change the csv\_filename location link to your own location that is storing the Metro\_Interstate\_Traffic\_Volume.csv



2\. Step 1: Run pipeline.py first to generate cleaned\_traffic\_data.csv.

&#x20;  Step 2: Run feature\_engineering.py to use generated cleaned\_traffic\_data.csv in Step 1 to generate engineered\_traffic\_features.csv.

&#x20;  Step 3: Run visualizations.py that is using engineered\_traffic\_features.csv generated in Step 2.

&#x20;  Step 4: Run CLIapp.py that is using engineered\_traffic\_features.csv generated in Step 2.





**Architecture Tracking \& Logging Configuration**



1. This project implements a professional, centralized logging structure via Python’s native logging library. To completely satisfy project constraints, it avoids root logging errors by instantiating module-isolated logger handles (logging.getLogger(\_\_name\_\_)) across every file component. All tasks write and append their logs directly into a single, consolidated file: pipeline.log.
2. Logging Blueprint Formatter Layout

&#x20;  Every log event across all operational scripts is tracked via this strict, explicit blueprint layout:

&#x20;  %(asctime)s \[%(levelname)s] %(name)s: %(message)s

&#x20;  Example Output: 2026-10-31 09:17:11 \[INFO] visualizations: Successfully saved figure 1...



3\. Meaningful Level Assignments Used

&#x20;  DEBUG:   Fine-grained background variables and intermediate calculation values (such as mathematical min-max scaling constraints, Z-score standard deviation matrices, and

&#x20;           statistical quartile boundaries) useful exclusively for developer system auditing. These entries remain completely hidden unless the core logger level threshold is

&#x20;           explicitly dialed down to DEBUG.

&#x20;  INFO:    Documents normal, expected structural pipeline milestones, processing checkpoints, data file loading successes, function completions, and visual figure export paths.

&#x20;  WARNING: Tracks unpredicted but safely recoverable dataset anomalies, such as row deduplication extractions, unphysical 0 Kelvin temperature updates, or physically impossible

&#x20;           precipitation sensor overrides.

&#x20;  ERROR:   Preserves critical runtime script failures or corrupted file inputs, mapping a clean system stack trace to the log file before executing a graceful shutdown hook to

&#x20;           avoid a raw system crash.





**Safe Terminal Execution Guard Rails**



1. Every user interactive function inside cli\_app.py is guarded by an ingestion data validator. If a user supplies a malformed input—such as typing alphabetic characters into a threshold block or keying a corrupted date format—the interface blocks an unhandled terminal crash, fires a clear ERROR log containing user entry variables into pipeline.log, and prints a dynamic assistance trace window displaying valid dataset ranges on screen.

