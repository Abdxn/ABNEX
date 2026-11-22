
import os
import json
import asyncio
import tempfile
import subprocess
import urllib.parse
import wave
import io
import threading

import edge_tts
import imageio_ffmpeg
import uvicorn

from groq import Groq
from fastapi import FastAPI, Request, Response, HTTPException
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
    "en": "en-IN-NeerjaNeural"
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found!")

groq_client = Groq(api_key=GROQ_API_KEY)
app = FastAPI(title="ABNEX Cloud")


# ==============================
# AI PERSONALITY
# ==============================

history = []
history_lock = threading.Lock()


def system_prompt():
    return """
You are ABNEX, a friendly personal AI assistant.
Understand Hindi and English.
Reply in the same language as the user.
Be intelligent, helpful and natural.
Keep answers concise but useful.
Do not use markdown.

Return ONLY valid JSON:
{
  "reply": "Your answer",
  "servo": null
}

If the user explicitly asks to move the servo,
return an angle between 0 and 180.
Otherwise servo must be null.
"""


# ==============================
# GROQ AI
# ==============================

def ask_groq(user_text):
    global history

    with history_lock:
        messages = [
            {"role": "system", "content": system_prompt()}
        ]

        messages.extend(history[-10:])
        messages.append({
            "role": "user",
            "content": user_text
        })

        result = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_completion_tokens=400,
            response_format={"type": "json_object"}
        )

        content = result.choices[0].message.content
        data = json.loads(content)

        reply = str(data.get("reply", "I am listening.")).strip()
        servo = data.get("servo")

        if servo is not None:
            try:
                servo = max(0, min(180, int(servo)))
            except (ValueError, TypeError):
                servo = None

        history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply}
        ])

        history = history[-10:]

        return reply, servo


# ==============================
# TEXT CHAT
# ==============================

@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
        user_text = str(body.get("text", "")).strip()

        if not user_text:
            raise HTTPException(400, "Question cannot be empty")

        if len(user_text) > 2000:
            raise HTTPException(413, "Question is too long")

        reply, servo = await run_in_threadpool(
            ask_groq, user_text
        )

        return {
            "status": "success",
            "reply": reply,
            "servo": servo
        }

    except HTTPException:
        raise
    except Exception as error:
        print("TEXT AI ERROR:", repr(error))
        raise HTTPException(502, "AI connection failed")


# ==============================
# CLOUD SPEECH RECOGNITION
# Incoming audio: 16kHz, 16-bit, mono PCM
# ==============================

def transcribe_audio(audio_bytes):
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_bytes)

    wav_buffer.seek(0)

    result = groq_client.audio.transcriptions.create(
        file=("abnex_audio.wav", wav_buffer.read(), "audio/wav"),
        model=STT_MODEL,
        response_format="verbose_json"
    )

    text = str(getattr(result, "text", "")).strip()
    language = str(getattr(result, "language", "en")).lower()

    if language.startswith("hi"):
        language = "hi"
    else:
        language = "en"

    return text, language


# ==============================
# TEXT TO SPEECH
# ==============================

async def generate_speech(text, language):
    voice = VOICES["hi"] if language == "hi" else VOICES["en"]

    with tempfile.TemporaryDirectory() as folder:
        mp3_path = os.path.join(folder, "reply.mp3")
        pcm_path = os.path.join(folder, "reply.pcm")

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(mp3_path)

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run(
            [
                ffmpeg, "-y",
                "-i", mp3_path,
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                pcm_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True
        )

        with open(pcm_path, "rb") as audio_file:
            return audio_file.read()


# ==============================
# AUDIO PROCESSING
# ==============================

def process_audio(audio_bytes):
    if len(audio_bytes) > 10 * 1024 * 1024:
        raise ValueError("Audio is too large")

    print("Audio received:", len(audio_bytes), "bytes")

    user_text, language = transcribe_audio(audio_bytes)
    print("USER:", user_text or "(not understood)")

    if not user_text:
        reply = "मुझे समझ नहीं आया, कृपया दोबारा बोलिए।"
        servo = None
        language = "hi"
    else:
        reply, servo = ask_groq(user_text)

    pcm_audio = asyncio.run(
        generate_speech(reply, language)
    )

    return pcm_audio, reply, servo


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/health")
def health():
    return {
        "status": "ABNEX CLOUD server OK",
        "model": GROQ_MODEL,
        "speech_recognition": STT_MODEL,
        "groq_api_configured": True
    }


# ==============================
# AUDIO TALK ENDPOINT
# ==============================

@app.post("/talk")
async def talk(request: Request):
    audio_bytes = await request.body()

    if len(audio_bytes) < 8000:
        raise HTTPException(400, "Audio too short")

    try:
        pcm_audio, reply, servo = await run_in_threadpool(
            process_audio, audio_bytes
        )

        headers = {
            "X-Reply-Text": urllib.parse.quote(reply, safe=""),
            "X-Servo": str(servo) if servo is not None else "none"
        }

        return Response(
            content=pcm_audio,
            media_type="application/octet-stream",
            headers=headers
        )

    except Exception as error:
        print("AUDIO ERROR:", repr(error))
        raise HTTPException(502, "Audio processing failed")


# ==============================
# START SERVER
# ==============================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT
    )
