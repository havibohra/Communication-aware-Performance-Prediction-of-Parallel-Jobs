import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score, mean_absolute_percentage_error

data_path=('/home/hbohra/Downloads/surge_data.csv')

dataset=pd.read_csv(data_path)
# Remove any rows with missing values if necessary
dataset = dataset.dropna()
print("Data Read :)")
# Normalize the numeric columns
scaler = MinMaxScaler()
numeric_columns = ['TotalNodes','TotalCores','IN_PORT', 'Rx', 'EX_PORT','Tx', 'RequiredTime','Timestamp','ActualTime','JobID']
dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])

# Reset index after sorting
dataset = dataset.reset_index(drop=True)

# One-Hot encode categorical features using pandas get_dummies
df = pd.get_dummies(dataset , columns=['UserName', 'Jobname', 'Device'])

print("Shape of dataset after Scaling: ",df.shape)

# Create the LSTM model
model2 = Sequential()
model2.add(LSTM(units=64, input_shape=(None,df.shape[1]-1)))
model2.add(Dense(1))

# Compile the model
model2.compile(loss='mean_squared_error', optimizer='adam')

print("Printing Model Requirements >>>")
[print(i.shape, i.dtype) for i in model2.inputs]
[print(o.shape, o.dtype) for o in model2.outputs]
[print(l.name, l.input_shape, l.dtype) for l in model2.layers]

# Sort the DataFrame by timestamps
df_sorted = df.sort_values('Timestamp')

# Get unique timestamps
unique_timestamps = df_sorted['Timestamp'].unique()

# Calculate the split index
split_index = int(len(unique_timestamps) * 0.8)

# Get the timestamps for training and testing
train_timestamps = unique_timestamps[:split_index]
test_timestamps = unique_timestamps[split_index:]

# Filter the data based on the train and test timestamps
train_data = df_sorted[df_sorted['Timestamp'].isin(train_timestamps)]
test_data = df_sorted[df_sorted['Timestamp'].isin(test_timestamps)]

for timestamp in train_timestamps:
    segment_data = train_data[train_data['Timestamp'] == timestamp]

    # Extract the features and target for the segment
    segment_sequence = segment_data.drop(['ActualTime'], axis=1)
    segment_sequence = np.reshape(segment_sequence.values, (segment_sequence.shape[0],1,segment_sequence.shape[1]))
    segment_targets= segment_data['ActualTime']
    print(f"Working on {timestamp} ...")
    # Train on it
    model2.fit(segment_sequence, segment_targets, epochs=5)

test_data= test_data[test_data['ActualTime']!=0]
test_sequence = test_data.drop(['ActualTime'], axis=1).values
test_sequence = np.reshape(test_sequence, (test_sequence.shape[0],1,test_sequence.shape[1]))
test_targets  = test_data['ActualTime'].values

print("Starting Evaluation...")
# Evaluate the model on the test data
loss = model2.evaluate(test_sequence, test_targets)

print("Test Loss:", loss)

print("Starting Predictions...")
# Perform predictions on the test data
predictions = model2.predict(test_sequence)

# Print the shape of predictions
print("Predictions Shape:", predictions.shape)

mse = mean_squared_error(test_targets, predictions)
print('Mean Squared Error:', mse)

rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

mae = mean_absolute_error(test_targets, predictions)
print('Mean Absolute Error:', mae)

r2 = r2_score(test_targets, predictions)
print('R-squared Score:', r2)

mape = mean_absolute_percentage_error(test_targets, predictions)
print('Mean Absolute Percentage Error:', mape)
