
import os
import sys
from uuid import uuid4

import openai
import whisper
from dotenv import find_dotenv, load_dotenv
from pytube import YouTube

link = sys.argv[1]

yt = YouTube(link).streams.filter(only_audio=True).first()

print(f"[INFO] Found video: {yt.title}.")


uid = uuid4()
yt.download(filename=f'{uid}.mp4')

print(f"[INFO] Downloaded audio from video.")

command2wav = f"/Users/mikko/Applications/ffmpeg/ffmpeg -i {uid}.mp4 {uid}.wav"
os.system(f'{command2wav} >/dev/null 2>&1')

print(f"[INFO] Converted video to WAV file.")

model = whisper.load_model("base")
result = model.transcribe(f"{uid}.wav", fp16=False, language='English')

print(f"[INFO] Transcribed speech from video.")

load_dotenv(dotenv_path=find_dotenv())
speech = result['text']
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_messages(text):
    msgs = [
    {'role': 'system', 'content': 'You are a helpful assistant that summarizes a transcript of a video.'},
    {'role': 'user', 'content': f'Summarize the following transcript: {text}'}
    ]
    return msgs

response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=generate_messages(speech),
        )

print(f"[INFO] Generated summary from transcript.")

os.remove(f"{uid}.mp4")
os.remove(f"{uid}.wav")

print(f"[INFO] Cleaning up files.")


res = response['choices'][0]['message']['content']
print(res)

