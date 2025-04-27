import os
import pandas as pd
import pyarrow.parquet as pq

# Folder containing the Parquet files
folder_path = "/home/dips/Documents/datasets_lerobot/aloha_mobile_cabinet/data/chunk-000"
output_folder = "/home/dips/Documents/datasets_lerobot/aloha_mobile_cabinet/data/chunk-000-csv"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all files in the folder
parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]

# Read each Parquet file and save it as a CSV file
for file in parquet_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_parquet(file_path)
    
    # Generate the output CSV file path
    output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.csv")
    
    # Save the dataframe as a CSV file
    df.to_csv(output_file, index=False)

print(f"Converted {len(parquet_files)} Parquet files to CSV and saved in {output_folder}")