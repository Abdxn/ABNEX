
import os
import json
import asyncio
import tempfile
import subprocess
import urllib.parse
import wave
import io
import time

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

# ==============================
# BRIDGE STATE
# ==============================

bridge_lock = Lock()

bridge_state = {
    "device_event_id": 0,
    "device_event": "",

    "phone_message_id": 0,
    "phone_text": "",
    "phone_reply": "",

    "voice_level": 0.0,
    "voice_pitch": 0.0,
    "voice_timestamp": 0.0,
    "voice_active": False,
}

# ==============================
# CORS
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://abdxn.github.io"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# ==============================
# AI SYSTEM PROMPT
# ==============================

SYSTEM_PROMPT = """
You are ABNEX, a friendly personal AI assistant.

Understand Hindi, Hinglish and English.

Reply naturally and concisely in the same language
the user uses.

Return ONLY valid JSON in this exact shape:
{"reply":"your answer","servo":null}

Set servo to an integer from 0 to 180 only if
the user explicitly asks to move the servo.

Otherwise servo must be null.

Do not include markdown fences.
"""


# ==============================
# GROQ AI
# ==============================

def ask_groq(user_text):

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_text
            },
        ],

        temperature=0.6,
        max_completion_tokens=500,

        response_format={
            "type": "json_object"
        },
    )

    raw = completion.choices[0].message.content or "{}"

    try:
        data = json.loads(raw)

        reply = str(
            data.get("reply", "")
        ).strip()

        servo = data.get("servo")

        if not reply:
            reply = (
                "Sorry, I could not prepare a response. "
                "Please try again."
            )

        if (
            not isinstance(servo, int)
            or isinstance(servo, bool)
            or not 0 <= servo <= 180
        ):
            servo = None

        return reply, servo

    except (
        json.JSONDecodeError,
        TypeError,
        ValueError
    ):

        return (
            raw.strip() or
            "Sorry, I could not prepare a response.",
            None
        )


# ==============================
# SPEECH RECOGNITION
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

        file=(
            "abnex_audio.wav",
            wav_buffer.read()
        ),

        model=STT_MODEL,
        response_format="json",
        language="hi",
    )

    text = (
        getattr(result, "text", "") or ""
    ).strip()

    return text, "hi"


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

        mp3_path = os.path.join(
            folder,
            "reply.mp3"
        )

        pcm_path = os.path.join(
            folder,
            "reply.pcm"
        )

        communicate = edge_tts.Communicate(
            text,
            voice
        )

        await communicate.save(mp3_path)

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                mp3_path,
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                pcm_path,
            ],
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

    print(
        "Audio received:",
        len(audio_bytes),
        "bytes"
    )

    user_text, language = transcribe_audio(
        audio_bytes
    )

    print(
        "USER:",
        user_text or "(not understood)"
    )

    if not user_text:

        reply = (
            "मुझे समझ नहीं आया, "
            "कृपया दोबारा बोलिए।"
        )

        servo = None
        language = "hi"

    else:

        reply, servo = ask_groq(user_text)

    pcm_audio = asyncio.run(
        generate_speech(reply, language)
    )

    return pcm_audio, reply, servo


# ==============================
# TEXT CHAT ENDPOINT
# ==============================

@app.post("/ask")
async def ask(request: Request):

    try:

        body = await request.json()

        user_text = str(
            body.get("text", "")
        ).strip()

        if not user_text:
            raise HTTPException(
                status_code=400,
                detail="Text is required"
            )

        if len(user_text) > 2000:
            raise HTTPException(
                status_code=413,
                detail="Text is too long"
            )

        reply, servo = await run_in_threadpool(
            ask_groq,
            user_text
        )

        return {
            "status": "success",
            "reply": reply,
            "servo": servo
        }

    except HTTPException:
        raise

    except Exception as error:

        print("ASK ERROR:", repr(error))

        raise HTTPException(
            status_code=502,
            detail="ABNEX could not generate a reply"
        )


# ==============================
# PHONE -> ESP32 DEVICE EVENTS
# ==============================

@app.post("/bridge/device-event")
async def bridge_device_event(request: Request):

    body = await request.json()

    event = str(
        body.get("event", "")
    ).strip()

    if event not in (
        "ptt_start",
        "ptt_stop"
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid device event"
        )

    with bridge_lock:

        bridge_state["device_event_id"] += 1

        bridge_state["device_event"] = event

        event_id = bridge_state["device_event_id"]

    return {
        "status": "ok",
        "event_id": event_id
    }


# ==============================
# PHONE POLL
# ==============================

@app.get("/bridge/phone-poll")
async def bridge_phone_poll(request: Request):

    try:

        after = int(
            request.query_params.get(
                "after",
                "0"
            )
        )

    except ValueError:
        after = 0

    with bridge_lock:

        changed = (
            bridge_state["device_event_id"] > after
        )

        return {

            "event_id":
                bridge_state["device_event_id"],

            "event":
                bridge_state["device_event"]
                if changed else ""
        }


# ==============================
# PHONE SENDS AI REPLY
# ==============================

@app.post("/bridge/phone-message")
async def bridge_phone_message(request: Request):

    body = await request.json()

    user_text = str(
        body.get("text", "")
    ).strip()[:2000]

    reply = str(
        body.get("reply", "")
    ).strip()[:4000]

    if not reply:

        raise HTTPException(
            status_code=400,
            detail="Reply is required"
        )

    with bridge_lock:

        bridge_state["phone_message_id"] += 1

        bridge_state["phone_text"] = user_text
        bridge_state["phone_reply"] = reply

        message_id = bridge_state["phone_message_id"]

    return {
        "status": "ok",
        "message_id": message_id
    }


# ==============================
# PHONE VOICE TELEMETRY
# ==============================

@app.post("/bridge/voice-level")
async def bridge_voice_level(request: Request):

    try:

        body = await request.json()

        level = float(
            body.get("level", 0)
        )

        pitch = float(
            body.get("pitch", 0)
        )

        active = bool(
            body.get("active", False)
        )

        if not (
            0.0 <= level <= 1.0
            and 0.0 <= pitch <= 1000.0
        ):

            raise HTTPException(
                status_code=400,
                detail="Invalid voice telemetry"
            )

        with bridge_lock:

            bridge_state["voice_level"] = level
            bridge_state["voice_pitch"] = pitch

            bridge_state["voice_timestamp"] = time.time()
            bridge_state["voice_active"] = active

        return {
            "status": "ok"
        }

    except HTTPException:
        raise

    except (
        ValueError,
        TypeError,
        json.JSONDecodeError
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid telemetry data"
        )


# ==============================
# ESP32 POLL
# ==============================

@app.get("/bridge/device-poll")
async def bridge_device_poll(request: Request):

    try:

        after = int(
            request.query_params.get(
                "after",
                "0"
            )
        )

    except ValueError:
        after = 0

    with bridge_lock:

        changed = (
            bridge_state["phone_message_id"] > after
        )

        # Voice telemetry expires after 1 second.
        telemetry_fresh = (
            time.time() -
            bridge_state["voice_timestamp"]
        ) < 1.0

        if telemetry_fresh:

            voice_level = bridge_state["voice_level"]
            voice_pitch = bridge_state["voice_pitch"]

        else:

            voice_level = 0.0
            voice_pitch = 0.0

        return {

            "message_id":
                bridge_state["phone_message_id"],

            "text":
                bridge_state["phone_text"]
                if changed else "",

            "reply":
                bridge_state["phone_reply"]
                if changed else "",

            "voice_level":
                voice_level,

            "voice_pitch":
                voice_pitch,

            "voice_active":
                bridge_state["voice_active"]
                if telemetry_fresh else False
        }


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/health")
def health():

    return {

        "status":
            "ABNEX CLOUD server OK",

        "model":
            GROQ_MODEL,

        "speech_recognition":
            STT_MODEL,

        "groq_api_configured":
            True,

        "voice_telemetry":
            True
    }


# ==============================
# AUDIO TALK ENDPOINT
# ==============================

@app.post("/talk")
async def talk(request: Request):

    audio_bytes = await request.body()

    if len(audio_bytes) < 8000:

        raise HTTPException(
            status_code=400,
            detail="Audio too short"
        )

    try:

        pcm_audio, reply, servo = await run_in_threadpool(
            process_audio,
            audio_bytes
        )

        headers = {

            "X-Reply-Text":
                urllib.parse.quote(
                    reply,
                    safe=""
                ),

            "X-Servo":
                str(servo)
                if servo is not None
                else "none",
        }

        return Response(
            content=pcm_audio,
            media_type="application/octet-stream",
            headers=headers
        )

    except Exception as error:

        print("AUDIO ERROR:", repr(error))

        raise HTTPException(
            status_code=502,
            detail="Audio processing failed"
        )


# ==============================
# START SERVER
# ==============================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT
    )
