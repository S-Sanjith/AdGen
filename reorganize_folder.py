# Reorganize directory structure to have each video and JSON inside its corresponding subfolder

import os
import shutil

def restructure_folders(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".mp4") or file.endswith(".json"):
                # Construct the full file path
                file_path = os.path.join(root, file)
                
                # Determine the new directory based on the file name
                new_dir = os.path.join(root, file.split('.')[0])
                
                # Create the new directory if it doesn't exist
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                
                # Move the file to the new directory
                shutil.move(file_path, os.path.join(new_dir, file))

if __name__ == "__main__":
    base_directory = "./test_data"  # folder to be reorganized
    restructure_folders(base_directory)