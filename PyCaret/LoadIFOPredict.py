from pycaret.anomaly import *
import pandas as pd

# Read in Datasets
# SNORT Logs
df = pd.read_csv(f"./Dataset/SNORT_COMBINED.csv")

# Panda Options to View All Rows & Columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option("expand_frame_repr", False)

# This loads the Trained IFO Model & Configuration
load_config("./Models/ifoConfig")
ifo = load_model("./Models/ifo-model-SNORT")

# This predicts "Anomalies" within dataset & displays all "Anomaly" data
predictions = predict_model(ifo, df)
print(predictions[predictions["Anomaly"] == 1])

# This plots an UMAP graph representation of the data
plot_model(ifo, plot='umap')

