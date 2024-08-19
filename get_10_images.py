import os
import subprocess

def extract_images_from_videos(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                # Define the output image pattern
                output_pattern = os.path.join(root, "%04d.png")
                
                # Construct the ffmpeg command
                command = [
                    'ffmpeg',
                    '-sseof', '-10',  # Start extraction from the last 10 seconds
                    '-i', video_path,  # Input video file
                    '-vf', 'fps=1',    # Extract 1 frame per second
                    output_pattern     # Output pattern for images
                ]
                
                # Execute the command
                subprocess.run(command, check=True)

if __name__ == "__main__":
    base_directory = "./test_data"  # Replace with your base directory
    extract_images_from_videos(base_directory)
