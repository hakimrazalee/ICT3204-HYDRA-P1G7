from pycaret.anomaly import *
import pandas as pd

# Read in Datasets
# SNORT Logs
df = pd.read_csv(f"./Dataset/SNORT_COMBINED.csv")

# Train <ML Technique> Model with Dataset & Identify "Anomalies" within Dataset with Labels
# Data Preprocessing -> Missing Value Imputation, One-Hot Encoding, Train-Test Split
#   Missing Value Imputation: Numerical -> Mean | Categorical -> not_available
#   One-Hot Encoding: Transforms Categorical features into numeric values
#   Train-Test Split: 7:3 Ratio (Train | Validation)
exp = setup(data=df, silent=True, use_gpu=True)
ifo = create_model('iforest', fraction=0.025)

# This is to export the trained SVM Model & Configurations for future predictions
save_model(ifo, "./Models/ifo-model-SNORT")
save_config("./Models/ifoConfig")
