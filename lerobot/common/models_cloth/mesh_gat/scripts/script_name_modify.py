import os
import re

# Define folder path
folder = "/home/ktang/ws/data/mesh_gat/t_shirt_l3/test_real"

# Loop through all files in the folder
for filename in os.listdir(folder):
    match = re.match(r"depth_(\d+)\.png", filename)
    if match:
        num = match.group(1)
        new_filename = f"{num}.depth.png"
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_filename)
        
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

print("Renaming complete!")