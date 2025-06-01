import streamlit as st
import joblib
import torch
import torchaudio
import numpy as np
from speechbrain.pretrained import EncoderClassifier
import tempfile
import os
from pydub import AudioSegment
import time

# Set the page config for Streamlit
st.set_page_config(page_title="Accent Classifier", layout="centered")
st.title("🎤 Accent Classifier App")
st.markdown("Upload an audio file and I'll predict the speaker's accent!")

# Set the torchaudio backend to "soundfile" for better compatibility on Streamlit Cloud
torchaudio.set_audio_backend("soundfile")

# Optionally, debug the paths for ffmpeg/ffprobe with pydub if needed:
from pydub.utils import which
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

@st.cache_resource
def load_model():
    return joblib.load("accent_classifier_model.pkl")

@st.cache_resource
def load_speechbrain_classifier():
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )

loaded_model = load_model()
classifier = load_speechbrain_classifier()

def convert_to_wav(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file.flush()
        temp_path = tmp_file.name

    # Load audio with pydub and convert to mono 16kHz wav
    audio = AudioSegment.from_file(temp_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Export to a temporary wav file
    wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_path = wav_tmp.name
    wav_tmp.close()
    audio.export(wav_path, format="wav")

    # Cleanup temporary file
    os.unlink(temp_path)

    return wav_path

def predict_accent(file_path):
    try:
        signal, fs = torchaudio.load(file_path)

        # If the sample rate isn't 16kHz, resample the signal.
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)

        # If the signal has multiple channels, convert to mono.
        if signal.ndim > 1 and signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0).unsqueeze(0)
        elif signal.ndim == 1:
            signal = signal.unsqueeze(0)

        with torch.no_grad():
            emb = classifier.encode_batch(signal)

        emb_np = emb.squeeze().cpu().numpy().reshape(1, -1)

        pred = loaded_model.predict(emb_np)[0]
        probs = loaded_model.predict_proba(emb_np)[0]
        confidence = np.max(probs)

        return pred, confidence

    except RuntimeError as e:
        # In case of backend-specific issues, try again after a slight delay.
        time.sleep(1)
        signal, fs = torchaudio.load(file_path)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)
        if signal.ndim > 1 and signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0).unsqueeze(0)
        elif signal.ndim == 1:
            signal = signal.unsqueeze(0)
        with torch.no_grad():
            emb = classifier.encode_batch(signal)
        emb_np = emb.squeeze().cpu().numpy().reshape(1, -1)
        pred = loaded_model.predict(emb_np)[0]
        probs = loaded_model.predict_proba(emb_np)[0]
        confidence = np.max(probs)
        return pred, confidence

uploaded_file = st.file_uploader("Upload an audio file (.wav, .mp3, .flac, .mp4)", type=["wav", "mp3", "flac", "mp4"])

if uploaded_file:
    st.audio(uploaded_file)

    with st.spinner("Analyzing audio..."):
        wav_path = convert_to_wav(uploaded_file)
        try:
            pred_accent, conf = predict_accent(wav_path)
            st.success(f"🎯 Predicted Accent: **{pred_accent}**")
            st.progress(int(conf * 100))
            st.write(f"🧠 Confidence: **{conf * 100:.2f}%**")
        except Exception as e:
            st.error(f"❌ {e}")
        finally:
            # Clean up the temporary WAV file
            os.remove(wav_path)
