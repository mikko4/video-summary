import os
import sys
from uuid import uuid4

import openai
import whisper
from dotenv import find_dotenv, load_dotenv
from pytube import YouTube

link = sys.argv[1]

yt = YouTube(link).streams.filter(only_audio=True).first()

print(f"[INFO] Found video: {yt.title}")


uid = uuid4()
yt.download(filename=f'{uid}.mp4')

command2wav = f"/Users/mikko/Applications/ffmpeg/ffmpeg -i {uid}.mp4 {uid}.wav"
os.system(f'{command2wav} >/dev/null 2>&1')

print(f"[INFO] Converted video to WAV file")

model = whisper.load_model("base")
print(f"[INFO] Transcribing video")

result = model.transcribe(f"{uid}.wav", fp16=False, language='English')

print(f"[INFO] Transcribed speech from video.")

load_dotenv(dotenv_path=find_dotenv())
speech = result['text']
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_messages(text):
    # estimate number of tokens in the text. according to openai, 1 token is approximately 0.75 words.
    tokens = len(text) * 4
    print(len(text))

    # trim the text unfortunately mate. This bot should work fully for most videos under 25mins in length but might have to cut short if longer.
    if tokens > 4000:
        text = text[:16000]
        print(len(text))

    msgs = [
    {'role': 'system', 'content': f"""
    You are a helpful assistant that summarizes a transcript of a video.
    You provide a concise yet thorough summaries of the key points and takeaways from the provided transcript.
    You provide the summary in a concise bullet-point format. Here is the video's title for context: {yt.title}"""},
    {'role': 'user', 'content': f'Summarize the following video transcript: {text}.'}
    ]

    return msgs

response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=generate_messages(speech),
        )

print(f"[INFO] Generated summary from transcript")

os.remove(f"{uid}.mp4")
os.remove(f"{uid}.wav")

print(f"[INFO] Cleaning up files")

# print(response)
res = response['choices'][0]['message']['content']
print(res)

