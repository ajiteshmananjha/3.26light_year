import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the serialized model from the JSON file
with open('model.json', 'r') as json_file:
    model_data = json.load(json_file)

# Deserialize the model using pickle
loaded_model = pickle.loads(model_data['model'].encode('latin-1'))

# Load the new data for prediction
new_data = pd.read_csv("sample_test_data.csv")

# Preprocess the new data (scale it if needed)
columns_to_scale_new_data = ['MEAN_RR','MEDIAN_RR','LF_NU','HF_NU','HF_LF','SDRR_RMSSD_REL_RR','HF_PCT','HF','SDSD_REL_RR','RMSSD_REL_RR','higuci','LF_HF','VLF','TP','sampen','SKEW','SKEW_REL_RR']
standard_scaler_new_data = StandardScaler()
new_data[columns_to_scale_new_data] = standard_scaler_new_data.fit_transform(new_data[columns_to_scale_new_data])

# Select the relevant features
X_new_data = new_data[['MEAN_RR','MEDIAN_RR','LF_NU','HF_NU','HF_LF','SDRR_RMSSD_REL_RR','HF_PCT','HF','SDSD_REL_RR','RMSSD_REL_RR','higuci','LF_HF','VLF','TP','sampen','SKEW','SKEW_REL_RR']]

# Make predictions on the new data
predictions_new_data = loaded_model.predict(X_new_data)

# Print the predictions
print("Predictions on New Data:")
print(predictions_new_data)
