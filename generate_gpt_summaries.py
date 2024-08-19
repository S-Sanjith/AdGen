import openai
import json

openai.api_key = 'api key here'

def summarize_text(text):
    message = [{"role": "system", "content": f"Create a detailed, concise, grammatically correct and coherent summary of the video advertisement. Describe the sequence of events or actions in the video by mainly utilizing the Transcript. The sequence of actions taking place in the video should form the major part of the summary. Incorporate information about the environment, people, place of the ad using the Image and also use the Title while generating the summary. Summary should be of 40 words, not exceeding 45 words. Rely strictly on the provided text. Example: the provided Transcript is 'Go to that famous city, the one you always see in the movies. And when you're there, ask Nobu to fix you something nice. Well done, chef. Thank you. Hop on another plane and go someplace else. Rested? Then go play in the sand and sing along with 90,000 fans. Go ahead and ask for his autograph. His too. Go to the city of love and just try to resist that accent. Layover in Doha? Pick up something nice at Harrods. Then go east, far east, and take care of business. Then see for yourself if these are the best dumplings in the world. Looks like they are. Or maybe you just want to go someplace that doesn't feel like anyplace else. Then you should really go here. So, where do you want to go?'. Then the sequence of action to be captured in the summary is like 'The ad for Qatar Airways describes traveling to several places and exploring them. Go to a famous city and have good food. Then take another plane to a different place. Play in the beach sand. Sing with fans in a concert and take autographs. Travel again. Travel for business. Stop at famous places and have the best dumplings. '. Focus mainly on the sequence of events from Transcript. Second preference to the environment details from Image. Keep it crisp and under 40 words."},
               {"role": "user", "content": text}]
    temperature = 0.2
    frequency_penalty = 0.0

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=message,
        temperature=temperature,
        max_tokens=256,
        frequency_penalty=frequency_penalty
    )
    return response.choices[0].message["content"].strip()

# Function to generate summary using GPT-4 Turbo
def generate_summary_gpt(model, title, description, llava_generated_text, transcript_text):

    # Combine all text inputs for summarization
    text_content = f"Title: {title}\nDescription: {description}\n Ad context information from image: {llava_generated_text}\nTranscript Text: {transcript_text}\n"

    # Generate summary using GPT-4 Turbo
    summary = summarize_text(text_content)
    return summary

# Read merged.json and transcripts_mod.json and use them to call generate_summary_gpt
with open('llava.json', 'r') as file:
    data = json.load(file)

with open('whisper.json', 'r', encoding='utf-8') as file:
    transcripts = json.load(file)

with open('metadata.json', 'r', encoding='utf-8') as file:
    metadata = json.load(file)

gpt_output = {}

n = 0

import concurrent.futures

def process_key(key):
    title = metadata[key]["title"]
    description = metadata[key]["description"]
    llava_generated_text = data[key]
    transcript_text = transcripts[key]
    summary = generate_summary_gpt("gpt-4-turbo", title, description, llava_generated_text, transcript_text)
    return key, summary

with concurrent.futures.ProcessPoolExecutor() as executor:
    for key, summary in executor.map(process_key, data.keys()):
        gpt_output[key] = summary
        print(f"Processed video {key}")

# Write the GPT-4 Turbo generated summaries to a JSON file
with open('gpt_summaries.json', 'w', encoding='utf-8') as file:
    json.dump(gpt_output, file, indent=4, ensure_ascii=False)