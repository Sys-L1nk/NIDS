import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import glob

# 1. Automatically find all CSV files inside the 'raw_data' folder
# The glob library grabs every file matching the *.csv pattern
all_files = glob.glob("raw_data/*.csv")

print(f"Found {len(all_files)} files. Combining them now...")

# 2. Read each file and store them in a list
dataframes_list = []
for file in all_files:
    print(f"Loading {file}...")
    temp_df = pd.read_csv(file, encoding='latin1')
    dataframes_list.append(temp_df)

# 3. Concatenate all loaded files into one massive dataset
df = pd.concat(dataframes_list, ignore_index=True)
print(f"Combined dataset shape: {df.shape}")

# 4. Clean infinity and null values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# 5. Encode categorical (text) columns to numeric
encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# 6. Save the single, massive processed dataset
df.to_csv('cleaned_combined_dataset.csv', index=False)
print("Finished! Saved as 'cleaned_combined_dataset.csv'")