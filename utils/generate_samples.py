import os
import random
import shutil

source_folder = "../data/tmQMg/xyz"
target_folder = "samples_data"
num_samples = 10000

# List all files in the source folder
all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Sample without replacement
selected = random.sample(all_files, num_samples)

# Make sure the target folder exists
os.makedirs(target_folder, exist_ok=True)

# Copy the selected files
for f in selected:
    shutil.copy2(os.path.join(source_folder, f), os.path.join(target_folder, f))

print("Done")
