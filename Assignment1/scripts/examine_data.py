# examine_data.py - Place this in your scripts folder and run it
import pandas as pd
import os

data_folder = "Assignment1/data/"
sample_file = os.path.join(data_folder, "NSE_data_all_stocks_2007.csv")

# Read just the first few rows to see column names
df_sample = pd.read_csv(sample_file, nrows=5)
print("=== COLUMN NAMES ===")
print(df_sample.columns.tolist())
print("\n=== FIRST 3 ROWS ===")
print(df_sample.head(3))
print("\n=== COLUMN INFO ===")
print(df_sample.info())