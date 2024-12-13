import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load data
dataTrain = pd.read_csv("DATA_TRAIN.csv")

# Check data frame slicing
print(dataTrain.iloc[1:11, 1:])

# Reshape the data to match the input shape of the model
input_data = dataTrain.iloc[1:11, 1:].to_numpy()  # Convert to NumPy
#input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

# Load model
model = load_model("trained_lstm_model.h5")

# Make predictions
y_pred = model.predict(input_data)

print(y_pred, '\n')

# Fix indexing for accessing Open and Close columns on the 12th row
print(dataTrain.loc[12])