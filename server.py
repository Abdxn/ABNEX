
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
