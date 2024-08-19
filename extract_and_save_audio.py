import os
import json
import yt_dlp

def download_audio_from_json(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)
                
                # Load the JSON data
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract the YouTube URL
                video_url = data.get('url')
                if video_url:
                    # Set the output filename to match the JSON filename (but with .mp3 extension)
                    mp3_filename = file.replace('.json', '')
                    mp3_output_path = os.path.join(root, mp3_filename)
                    
                    # Download the audio using yt-dlp
                    download_audio(video_url, mp3_output_path)

def download_audio(url, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    base_directory = "./test_data"  # base directory containing the videos
    download_audio_from_json(base_directory)
