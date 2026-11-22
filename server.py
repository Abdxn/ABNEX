
import os
import json
import asyncio
import subprocess
import tempfile
import urllib.parse
import time

import numpy as np
import edge_tts
import uvicorn

from groq import Groq
from faster_whisper import WhisperModel
from fastapi import FastAPI, Request, Response, HTTPException
from starlette.concurrency import run_in_threadpool


# ==============================
# ABNEX V7 SERVER
# ==============================

PORT = 8000
GROQ_MODEL = "openai/gpt-oss-20b"
WHISPER_SIZE = "base"

VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "en": "en-IN-NeerjaNeural"
}

print("\nLoading ABNEX Whisper model...")

whisper = WhisperModel(
    WHISPER_SIZE,
    device="cpu",
    compute_type="int8"
)

print("Whisper ready!")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found!")

groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="ABNEX V7")


# ==============================
# AI PERSONALITY
# ==============================

history = []

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

    messages = [
        {
            "role": "system",
            "content": system_prompt()
        }
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

    data = json.loads(
        result.choices[0].message.content
    )

    reply = str(
        data.get("reply", "I am listening.")
    ).strip()

    servo = data.get("servo")

    if servo is not None:
        try:
            servo = max(0, min(180, int(servo)))
        except (ValueError, TypeError):
            servo = None

    history.append({
        "role": "user",
        "content": user_text
    })

    history.append({
        "role": "assistant",
        "content": reply
    })

    history = history[-10:]

    return reply, servo


# ==============================
# TEXT CHAT ENDPOINT
# ==============================

@app.post("/ask")
async def ask(request: Request):

    try:
        body = await request.json()
        user_text = str(body.get("text", "")).strip()

        if not user_text:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        if len(user_text) > 2000:
            raise HTTPException(
                status_code=413,
                detail="Question is too long"
            )

        print("\nTEXT QUESTION:", user_text)

        reply, servo = await run_in_threadpool(
            ask_groq,
            user_text
        )

        print("ABNEX:", reply)

        return {
            "status": "success",
            "reply": reply,
            "servo": servo
        }

    except HTTPException:
        raise

    except Exception as error:
        print("TEXT AI ERROR:", repr(error))

        raise HTTPException(
            status_code=502,
            detail="AI connection failed"
        )


# ==============================
# SPEECH RECOGNITION
# ==============================

def transcribe_audio(audio_bytes):

    audio = np.frombuffer(
        audio_bytes,
        dtype=np.int16
    )

    if len(audio) == 0:
        return "", "en"

    audio = audio.astype(np.float32) / 32768.0

    segments, info = whisper.transcribe(
        audio,
        beam_size=1,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 150
        }
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    return text, info.language or "en"


# ==============================
# TEXT TO SPEECH
# ==============================

async def generate_speech(text, language):

    voice = (
        VOICES["hi"]
        if language == "hi"
        else VOICES["en"]
    )

    with tempfile.TemporaryDirectory() as folder:

        mp3_path = os.path.join(folder, "reply.mp3")
        pcm_path = os.path.join(folder, "reply.pcm")

        communicate = edge_tts.Communicate(
            text,
            voice
        )

        await communicate.save(mp3_path)

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", mp3_path,
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", "16000",
                pcm_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        with open(pcm_path, "rb") as file:
            return file.read()


# ==============================
# PROCESS AUDIO
# ==============================

def process_audio(audio_bytes):

    total_start = time.time()

    print(
        f"\nAudio received: {len(audio_bytes)} bytes"
    )

    start = time.time()

    user_text, language = transcribe_audio(
        audio_bytes
    )

    print(
        f"Whisper time: {time.time() - start:.2f}s"
    )

    print("USER:", user_text or "(not understood)")

    if not user_text:

        reply = "मुझे समझ नहीं आया, कृपया दोबारा बोलिए।"
        servo = None
        language = "hi"

    else:

        try:
            start = time.time()

            reply, servo = ask_groq(user_text)

            print(
                f"Groq time: {time.time() - start:.2f}s"
            )

        except Exception as error:

            print("Groq error:", repr(error))

            reply = "माफ़ कीजिए, अभी AI से कनेक्शन में दिक्कत है।"
            servo = None
            language = "hi"

    try:

        start = time.time()

        pcm_audio = asyncio.run(
            generate_speech(reply, language)
        )

        print(
            f"TTS time: {time.time() - start:.2f}s"
        )

    except Exception as error:

        print("TTS error:", repr(error))
        pcm_audio = b""

    print(
        f"TOTAL: {time.time() - total_start:.2f}s"
    )

    return pcm_audio, reply, servo


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/health")
def health():

    return {
        "status": "ABNEX V7 server OK",
        "model": GROQ_MODEL,
        "whisper": WHISPER_SIZE,
        "groq_api_configured": True
    }


# ==============================
# AUDIO TALK ENDPOINT
# ==============================

@app.post("/talk")
async def talk(request: Request):

    audio_bytes = await request.body()

    if len(audio_bytes) < 8000:

        return Response(
            content="Audio too short",
            status_code=400
        )

    pcm_audio, reply, servo = await run_in_threadpool(
        process_audio,
        audio_bytes
    )

    headers = {
        "X-Reply-Text": urllib.parse.quote(
            reply,
            safe=""
        ),
        "X-Servo": (
            str(servo)
            if servo is not None
            else "none"
        )
    }

    return Response(
        content=pcm_audio,
        media_type="application/octet-stream",
        headers=headers
    )


# ==============================
# START SERVER
# ==============================

if __name__ == "__main__":

    print("\n==============================")
    print("       ABNEX V7 SERVER")
    print("==============================")
    print("AI Model:", GROQ_MODEL)
    print("Whisper:", WHISPER_SIZE)
    print("Text endpoint: /ask")
    print("Audio endpoint: /talk")
    print("Health endpoint: /health")
    print("==============================\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT
    )
