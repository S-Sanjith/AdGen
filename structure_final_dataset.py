import os
import shutil

def organize_dataset(base_dir, output_dir):
    # Define the output directories
    videos_dir = os.path.join(output_dir, "videos")
    images_dir = os.path.join(output_dir, "images")
    
    # Create the output directories if they don't exist
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # Walk through the base directory
    for root, dirs, files in os.walk(base_dir):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            for file in os.listdir(subdir_path):
                if file.endswith(".mp4"):
                    video_source = os.path.join(subdir_path, file)
                    video_destination = os.path.join(videos_dir, f"{subdir}.mp4")
                    shutil.copy(video_source, video_destination)
                
                if file == "best_image.png":
                    image_source = os.path.join(subdir_path, file)
                    image_destination = os.path.join(images_dir, f"{subdir}.png")
                    shutil.copy(image_source, image_destination)
    
    # Copy gpt_summaries.json to text_captions.json
    gpt_summaries_path = './gpt_summaries.json'
    text_captions_path = os.path.join(output_dir, 'text_captions.json')
    if os.path.exists(gpt_summaries_path):
        shutil.copy(gpt_summaries_path, text_captions_path)

if __name__ == "__main__":
    base_directory = "test_data"  
    dataset_directory = "dataset"  # Directory to store the final dataset
    organize_dataset(base_directory, dataset_directory)
