import os
import random

DATASET_DIR = './data/not_wakeword_dataset/'
OUTPUT_DIR = './data/not_wakeword_from_dataset/'

def load_datasets():
    # Because we intend for the user to download the entire speech_commands dataset,
    # but don't want all the data, we will only load a random 10000 samples.
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for filename in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed file {file_path}")
    
    # List every file from every subfolder in the dataset directory
    files = []
    for root, dirs, filenames in os.walk(DATASET_DIR):
        for filename in filenames:
            if filename.endswith('.wav'):
                files.append(os.path.join(root, filename))
    
    # Randomly select 10000 files
    selected_files = random.sample(files, 10000)
    
    # Copy the random files to the top-level dataset directory
    for i, file in enumerate(selected_files):
        new_file_path = os.path.join(OUTPUT_DIR, f"not_wakeword_{i}.wav")
        if not os.path.exists(new_file_path):
            os.rename(file, new_file_path)
            print(f"Moved {file} to {new_file_path}")
        else:
            print(f"File {new_file_path} already exists, skipping.")