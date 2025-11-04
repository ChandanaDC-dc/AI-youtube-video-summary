# app.py
# 🎧 YouTube → Audio → Text → Summary Web App
# Run using: streamlit run app.py

import os
import re
import textwrap
import requests
import yt_dlp
import whisper
import streamlit as st

# =============================
# CONFIGURATION
# =============================

HF_API_KEY = "your hugging face api key"
MODEL_NAME = "facebook/bart-large-cnn"

# =============================
# AUDIO DOWNLOAD
# =============================

def download_audio(video_url, filename="audio.wav"):
    video_url = video_url.split("?")[0]

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'noplaylist': True,
        'nocheckcertificate': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # Fix double extension issue
    if os.path.exists("audio.wav"):
        return "audio.wav"
    elif os.path.exists("audio.wav.wav"):
        os.rename("audio.wav.wav", "audio.wav")
        return "audio.wav"
    else:
        return None

# =============================
# TRANSCRIBE AUDIO (WHISPER)
# =============================

def transcribe_audio(audio_file="audio.wav"):
    if not os.path.exists(audio_file):
        st.error("❌ Audio file not found. Please download first.")
        return ""

    st.info("🧠 Loading Whisper model (small)…")
    model = whisper.load_model("small")

    st.info("🎙️ Transcribing audio… please wait…")
    result = model.transcribe(audio_file)
    text = result["text"]

    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(text)

    return text

# =============================
# SUMMARIZATION (HUGGING FACE)
# =============================

def clean_summary(text):
    cleaned = re.sub(r'\b(\w+\s+){1,3}\1+', r'\1', text)
    cleaned = re.sub(r'(\b\w+\b)( \1\b)+', r'\1', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def summarize_text(text, chunk_size=1000):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    summaries = []
    chunks = textwrap.wrap(text, chunk_size, break_long_words=False, replace_whitespace=False)

    for i, chunk in enumerate(chunks, 1):
        st.write(f"⏳ Summarizing chunk {i}/{len(chunks)}...")
        payload = {"inputs": chunk, "parameters": {"min_length": 100, "max_length": 300, "temperature": 0.7}}
        response = requests.post(f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}",
                                 headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and "summary_text" in data[0]:
                summary = clean_summary(data[0]["summary_text"])
                summaries.append(summary)
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")

    final_summary = clean_summary(" ".join(summaries))
    return final_summary

# =============================
# STREAMLIT WEB UI
# =============================

st.set_page_config(page_title="🎧 YouTube AI Summarizer", page_icon="🎙️", layout="wide")

st.title("🎧 YouTube AI Summarizer")
st.markdown("Convert YouTube videos → Text → Smart AI Summary 🧠")

video_url = st.text_input("🔗 Enter YouTube video URL:")

if st.button("1️⃣ Download Audio"):
    with st.spinner("Downloading audio..."):
        audio_path = download_audio(video_url)
        if audio_path:
            st.success("✅ Audio downloaded successfully!")
            st.audio(audio_path)
        else:
            st.error("❌ Failed to download audio.")

if st.button("2️⃣ Transcribe Audio"):
    with st.spinner("Transcribing..."):
        transcript = transcribe_audio()
        if transcript:
            st.success("✅ Transcription complete!")
            st.text_area("📝 Full Transcript:", transcript, height=200)

if st.button("3️⃣ Generate Summary"):
    if os.path.exists("transcript.txt"):
        with open("transcript.txt", "r", encoding="utf-8") as f:
            text = f.read()
        with st.spinner("Summarizing..."):
            summary = summarize_text(text)
            st.success("✅ Summary generated successfully!")
            st.text_area("🧾 AI Summary:", summary, height=200)
    else:
        st.error("⚠️ Transcript not found. Please transcribe first.")

