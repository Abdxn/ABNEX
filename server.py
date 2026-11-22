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
    text = (getattr(result, "text", "") or "").strip()
    # Keep Hindi/Hinglish default; English phrases can still be understood by the AI.
    return text, "hi"


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
            [ffmpeg, "-y", "-i", mp3_path, "-f", "s16le", "-acodec", "pcm_s16le",
             "-ac", "1", "-ar", str(SAMPLE_RATE), pcm_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
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
    pcm_audio = asyncio.run(generate_speech(reply, language))
    return pcm_audio, reply, servo


# ==============================
# TEXT CHAT ENDPOINT
# ==============================
@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
        user_text = str(body.get("text", "")).strip()
        if not user_text:
            raise HTTPException(status_code=400, detail="Text is required")
        if len(user_text) > 2000:
            raise HTTPException(status_code=413, detail="Text is too long")
        reply, servo = await run_in_threadpool(ask_groq, user_text)
        return {"status": "success", "reply": reply, "servo": servo}
    except HTTPException:
        raise
    except Exception as error:
        print("ASK ERROR:", repr(error))
        raise HTTPException(status_code=502, detail="ABNEX could not generate a reply")


# ==============================
# HEALTH CHECK
# ==============================
@app.get("/health")
def health():
    return {
        "status": "ABNEX CLOUD server OK",
        "model": GROQ_MODEL,
        "speech_recognition": STT_MODEL,
        "groq_api_configured": True,
    }


# ==============================
# AUDIO TALK ENDPOINT
# ==============================
@app.post("/talk")
async def talk(request: Request):
    audio_bytes = await request.body()
    if len(audio_bytes) < 8000:
        raise HTTPException(status_code=400, detail="Audio too short")
    try:
        pcm_audio, reply, servo = await run_in_threadpool(process_audio, audio_bytes)
        headers = {
            "X-Reply-Text": urllib.parse.quote(reply, safe=""),
            "X-Servo": str(servo) if servo is not None else "none",
        }
        return Response(content=pcm_audio, media_type="application/octet-stream", headers=headers)
    except Exception as error:
        print("AUDIO ERROR:", repr(error))
        raise HTTPException(status_code=502, detail="Audio processing failed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
