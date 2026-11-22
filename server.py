import os
import json
import asyncio
import tempfile
import subprocess
import urllib.parse
import wave
import io

import edge_tts
import imageio_ffmpeg
import uvicorn
from groq import Groq
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock
from starlette.concurrency import run_in_threadpool

# ==============================
# ABNEX CLOUD SERVER
# ==============================
PORT = int(os.getenv("PORT", "8000"))
GROQ_MODEL = "openai/gpt-oss-20b"
STT_MODEL = "whisper-large-v3-turbo"
SAMPLE_RATE = 16000
VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "en": "en-IN-NeerjaNeural",
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is missing")

groq_client = Groq(api_key=GROQ_API_KEY)
app = FastAPI(title="ABNEX Cloud")

# Single-device prototype bridge state (one Render instance).
bridge_lock = Lock()
bridge_state = {
    "device_event_id": 0, "device_event": "", "phone_message_id": 0,
    "phone_text": "", "phone_reply": ""
}

# Allow the ABNEX GitHub Pages voice gateway to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://abdxn.github.io"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

SYSTEM_PROMPT = """You are ABNEX, a friendly personal AI assistant. Understand Hindi, Hinglish and English. Reply naturally and concisely in the same language the user uses. Return ONLY valid JSON in this exact shape: {\"reply\":\"your answer\",\"servo\":null}. Set servo to an integer from 0 to 180 only if the user explicitly asks to move the servo; otherwise null. Do not include markdown fences."""


def ask_groq(user_text):
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0.6,
        max_completion_tokens=500,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        reply = str(data.get("reply", "")).strip()
        servo = data.get("servo")
        if not reply:
            reply = "Sorry, I could not prepare a response. Please try again."
        if not isinstance(servo, int) or isinstance(servo, bool) or not 0 <= servo <= 180:
            servo = None
        return reply, servo
    except (json.JSONDecodeError, TypeError, ValueError):
        return raw.strip() or "Sorry, I could not prepare a response.", None


def transcribe_audio(audio_bytes):
    # /talk receives 16 kHz, 16-bit, mono PCM from the ESP32.
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_bytes)
    wav_buffer.seek(0)
    result = groq_client.audio.transcriptions.create(
        file=("abnex_audio.wav", wav_buffer.read()),
        model=STT_MODEL,
        response_format="json",
        language="hi",
    )
