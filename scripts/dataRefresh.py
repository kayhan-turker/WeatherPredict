import os
from settings import *

folder_path = TRAINING_DATA_PATH
output_file = WEATHER_DATA_PATH + "raw_data.txt"

records = []

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):  # Ensure only processing relevant files
        parts = filename[:-4].split("_")  # Remove .jpg and split by _
        records.append(";".join(parts))

with open(output_file, "w") as f:
    f.write("\n".join(records))

print(f"Saved {len(records)} records to {output_file}")
