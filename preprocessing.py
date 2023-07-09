import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import re
import os

#Give path to data files
data_path=('/content/drive/My Drive/learning')

### Taking columns from counters_table files 
combined_df = pd.DataFrame()
for file_name in os.listdir(data_path):
    if file_name.startswith('counters_table_'):
        file_path = os.path.join(data_path, file_name)

        df= pd.read_csv(file_path,sep=' ', names=['Device','Port','Tx','Rx','Timestamp'],index_col=False )
        combined_df = pd.concat([combined_df, df])
print(combined_df)

### Taking columns from jobs_table files 
job_profile= pd.DataFrame()
for file_name in os.listdir(data_path):
    if file_name.startswith('jobs_table_'):
        timestamp = file_name[-16:-4]
        file_path = os.path.join(data_path, file_name)

        df= pd.read_csv(file_path,sep=' ', names=['JobID','UserName','QueueName','TotalNodes','TotalCores','RequiredTime','JobState','ElaspedTime','NodeList'],index_col=False,header=1 )
        df['Timestamp']=timestamp
        df = df.drop(['QueueName','JobState','ElaspedTime','NodeList'],axis=1)

        job_profile= pd.concat([job_profile, df])

job_profile = job_profile.set_index('Timestamp')
print(job_profile)


### Taking columns from paths_table files 
paths_data= pd.DataFrame()
for file_name in os.listdir(data_path):
    if file_name.startswith('paths_table_'):
        timestamp = file_name[-16:-4]
        file_path = os.path.join(data_path, file_name)

        with open(file_path, 'r') as file:
          lines = file.readlines()
        # Process each line and append the results to the DataFrame
        output_df = pd.DataFrame(columns=['JobID', 'Device', 'IN_PORT', 'EX_PORT'])

        # Process each line and append the results to the DataFrame
        for line in lines:
            # Split the line into columns
            columns = line.split()

            # Split the third column into multiple parts
            parts = columns[2].split('->')

            # Extract the desired values and create a temporary DataFrame
            # temp_df = pd.DataFrame(columns=['File', 'Device', 'IN_PORT', 'EX_PORT'])
            skip_next = False
            for i in range(len(parts)):
                if skip_next:
                    skip_next = False
                    continue
                device_name = parts[i].split(':')[0]
                port = int(parts[i].split(':')[1])

                if device_name.startswith('hpc'):
                    in_port = port
                    ex_port = port
                else:
                    in_port = port
                    ex_port = int(parts[i+1].split(':')[1])
                    skip_next = True

                output_df = pd.concat([output_df, pd.DataFrame({'JobID': [columns[1]], 'Device': [device_name], 'IN_PORT': [in_port], 'EX_PORT': [ex_port]})], ignore_index=True)

        output_df = output_df.drop_duplicates()
        output_df['Timestamp']=timestamp
        paths_data= pd.concat([paths_data, output_df])
# paths_data
paths_data= paths_data.drop_duplicates()
print(paths_data)
# Remove any rows with missing values if necessary
paths_data = paths_data.dropna()

### paths_data.to_csv('D:/p_data.csv', index=False) ---- save file if required

### Merging paths_data with job_profile data and then with counter_table data

paths_data['IN_PORT']=paths_data['IN_PORT'].astype('int64')
paths_data['EX_PORT']=paths_data['EX_PORT'].astype('int64')

merged_df = pd.merge(job_profile, paths_data, on=['Timestamp', 'JobID'], how='left')
merged_df['Timestamp'] = merged_df['Timestamp'].astype('int64')
# merged_df['IN_PORT'] = merged_df['IN_PORT'].astype('int64')
# merged_df['EX_PORT'] = merged_df['EX_PORT'].astype('int64')

combined_df['Port'] = combined_df['Port'].astype('int64')
combined_df   = combined_df.rename(columns={'Port':'IN_PORT' })

merged_df_final = pd.merge(merged_df, combined_df, on=['Timestamp', 'Device', 'IN_PORT'], how='left')
merged_df_final= merged_df_final.drop('Tx',axis=1);

combined_df   = combined_df.rename(columns={'IN_PORT' : 'EX_PORT', 'Rx':'Rx2'})
merged_df_final2 = pd.merge(merged_df_final, combined_df, on=['Timestamp', 'Device', 'EX_PORT'], how='left')
merged_df_final2= merged_df_final2.drop('Rx2',axis=1);

print(merged_df_final2)

### merged_df_final2.to_csv('D:/md_data.csv', index=False) ---- save if required

### Collecting job names
job_names_df = pd.DataFrame()
for file_name in os.listdir(data_path):
    if file_name.startswith('qstat_data_'):
        file_path = os.path.join(data_path, file_name)

        df= pd.read_csv(file_path,delim_whitespace=True, names=['JobID','Username','Queue','Jobname','SessID','NDS','TSK','reqdm','reqt','s','elt','nodes'],index_col=False ,header=4 )
        df=df.drop(['Username','Queue','SessID','NDS','TSK','reqdm','reqt','s','elt','nodes'],axis=1)
        job_names_df = pd.concat([job_names_df, df])
job_names_df =job_names_df.drop_duplicates()
print(job_names_df)

### merging job names
final_dataset = pd.merge(merged_df_final2, job_names_df, on=['JobID'], how='left')
final_dataset= final_dataset[['Timestamp','JobID','UserName','Jobname','TotalNodes','TotalCores','RequiredTime','Device','IN_PORT','Rx','EX_PORT','Tx']]
final_dataset['JobID'] = final_dataset['JobID'].str.split('.').str[0]
final_dataset

# final_dataset.to_csv('D:/all_data.csv', index=False) --- save if required

# final_dataset=pd.read_csv(data_path)

# final_dataset


########### SOME OF PREPROCESSING IS DONE HERE ########## 

# Remove rows with value 2 in the "TotalNodes" column
final_dataset = final_dataset[final_dataset['TotalNodes'] != 2]
print(final_dataset)

## Conversion Of Req. time to minutes ##
# Split 'RequiredTime' column into hours and minutes
final_dataset[['Hours', 'Minutes']] = final_dataset['RequiredTime'].str.split(':', expand=True)

# Convert hours and minutes to numeric format
final_dataset['Hours'] = pd.to_numeric(final_dataset['Hours'])
final_dataset['Minutes'] = pd.to_numeric(final_dataset['Minutes'])

# Calculate the total time in minutes
final_dataset['RequiredTime'] = final_dataset['Hours'] * 60 + final_dataset['Minutes']

# Drop the 'Hours' and 'Minutes' columns
final_dataset = final_dataset.drop(['Hours', 'Minutes'], axis=1)

# Convert Timestamp column to string type and add leading zeros
final_dataset['Timestamp'] = final_dataset['Timestamp'].astype(str)
final_dataset['Timestamp'] = final_dataset['Timestamp'].str.zfill(12)

from datetime import datetime, timedelta

# Assuming 'timestamp_str' is the string column containing the timestamps in the format "DDMMYYYYHHMM"
timestamps = final_dataset['Timestamp']

# Define the input format of the timestamp string
input_format = "%d%m%Y%H%M"

# Convert the timestamp string to datetime objects
datetimes = [datetime.strptime(timestamp, input_format) for timestamp in timestamps]

# Calculate the time difference from a reference point (e.g., the start of the dataset)
reference_time = datetimes[1314130]  # Assuming the given datetime index is the reference point (I adjusted the row no. as per some error in sorting)
timedeltas = [timestamp - reference_time for timestamp in datetimes]

# Convert the timedeltas to total seconds
total_seconds = [timedelta.total_seconds() for timedelta in timedeltas]

# Update the dataframe column with the timedeltas or total seconds
final_dataset['Timestamp'] = timedeltas  # Or df['timestamp'] = total_seconds

# final_dataset

final_dataset['Timestamp'] = pd.to_timedelta(final_dataset['Timestamp']).dt.total_seconds()
print(final_dataset)

### CHECK IF DTYPES ARE AS REQUIRED
print(final_dataset['Timestamp'].dtype) #datetime64ns
print(final_dataset['JobID'].dtype)
print(final_dataset['UserName'].dtype) #string
print(final_dataset['Jobname'].dtype) #string
print(final_dataset['TotalNodes'].dtype)
print(final_dataset['TotalCores'].dtype)
print(final_dataset['RequiredTime'].dtype) #in minutes
print(final_dataset['Device'].dtype) #string
print(final_dataset['IN_PORT'].dtype)
print(final_dataset['Rx'].dtype)
print(final_dataset['EX_PORT'].dtype)
print(final_dataset['Tx'].dtype)

final_dataset['Timestamp'] = final_dataset['Timestamp'].astype(int)

# Remove any rows with missing values if necessary
final_dataset = final_dataset.dropna()

print(final_dataset)

# final_dataset.to_csv('/content/drive/My Drive/learning/data_without_actualtime.csv', index=False) ---SAVE IF REQUIRED


####################### SAVE THE PREPARED DATASET TO A LOCATION ##################
final_dataset.to_csv('/content/drive/My Drive/learning/prep_data.csv', index=False)

#---><---#