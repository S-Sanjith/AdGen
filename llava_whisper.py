import os
import json
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import whisper

# Load the image classification model (e.g., ResNet50)
def load_image_classification_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.eval()
    return model

# Load the LLava model with 4-bit quantization
def load_llava_model(model_id):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(model_id)
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=quantization_config, 
        device_map="auto"
    )
    return processor, llava_model

# Load Whisper model for audio transcription
def load_whisper_model(model_size="small"):
    return whisper.load_model(model_size)

# Preprocessing transforms for images
def get_image_preprocessing_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Extract relevant information from a JSON file
def extract_json_info(json_file_path):
    if not os.path.exists(json_file_path):
        return "", "", []
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    info = data.get('yt_meta_dict', {}).get('info', {})
    title = info.get('title', "")
    description = info.get('description', "")
    tags = [tag.encode('ascii', 'ignore').decode() for tag in info.get('tags', [])]
    
    return title, description, tags

# Tokenize text into lowercase words
def tokenize_text(text):
    return text.lower().split()

# Calculate relevance score based on the model's logits and tokenized text
def calculate_relevance(logits, tokenized_text):
    return logits.sum().item()

# Select the best image based on relevance to the provided description
def select_best_image(image_files, text_description, model, preprocess):
    best_image = None
    best_score = float('-inf')
    tokenized_text = tokenize_text(text_description)
    
    for image_file in image_files:
        image = Image.open(image_file)
        with torch.no_grad():
            logits = model(preprocess(image).unsqueeze(0))
        relevance_score = calculate_relevance(logits, tokenized_text)
        
        if relevance_score > best_score:
            best_score = relevance_score
            best_image = image_file
    
    if best_image:
        # Extract file extension
        _, extension = os.path.splitext(best_image)
        new_name = os.path.join(os.path.dirname(best_image), f"best_image{extension}")
        os.rename(best_image, new_name)
        best_image = new_name
    
    return best_image

# Main processing loop for the dataset folders
def process_dataset_folders(data_path, model, preprocess, processor, llava_model, whisper_model):
    llava_texts = {}
    whisper_output = {}
    metadata = {}

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            json_file = os.path.join(subfolder_path, f"{subfolder}.json")
            title, description, tags = extract_json_info(json_file)

            # Save metadata
            metadata[subfolder] = {
                "title": title,
                "description": description,
                "tags": tags
            }

            # Select the best image from the subfolder
            image_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            best_image_path = select_best_image(image_files, description, model, preprocess)

            # Generate text using LLava model
            if best_image_path:
                image1 = Image.open(best_image_path)
                prompts = [
                    f"USER: <image>\nCreate a coherent text description by utilizing the image and capturing every minute detail from the image, its context, environment, people, place, colors, objects, animals, logo, product etc in detail. Capture all the essential ideas, vital information from the image and give an in-depth description. Additionally, recognize any famous people names, product logo, product taglines, vital objects, famous place names and most importantly any product. \nASSISTANT: "
                ]
                inputs = processor(prompts, images=[image1], padding=True, return_tensors="pt").to("cuda")
                output = llava_model.generate(**inputs, max_new_tokens=200)
                llava_generated_text = processor.batch_decode(output, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()
                llava_texts[subfolder] = llava_generated_text

            # Transcribe the audio file using Whisper
            audio_path = os.path.join(subfolder_path, f"{subfolder}.mp3")
            if os.path.exists(audio_path):
                result = whisper_model.transcribe(audio_path)
                whisper_output[subfolder] = result["text"]

    return llava_texts, whisper_output, metadata

# Save the generated outputs to JSON files
def save_to_json(data, file_path):
    with open(file_path, "w", encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

def main():
    # Load models and preprocessors
    model = load_image_classification_model()
    processor, llava_model = load_llava_model("llava-hf/llava-1.5-13b-hf")
    whisper_model = load_whisper_model()
    preprocess = get_image_preprocessing_transforms()

    # Define paths
    data_path = "./test_data"
    llava_path = "llava.json"
    whisper_path = "whisper.json"
    metadata_path = "metadata.json"

    # Process dataset folders and generate outputs
    llava_texts, whisper_output, metadata = process_dataset_folders(
        data_path, model, preprocess, processor, llava_model, whisper_model
    )

    # Save outputs to JSON files
    save_to_json(llava_texts, llava_path)
    save_to_json(whisper_output, whisper_path)
    save_to_json(metadata, metadata_path)

if __name__ == "__main__":
    main()
