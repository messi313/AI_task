import streamlit as st
import joblib
import torch
import torchaudio
import numpy as np
from speechbrain.pretrained import EncoderClassifier
import tempfile
import os
from pydub import AudioSegment
import subprocess
import time

st.set_page_config(page_title="Accent Classifier", layout="centered")
st.title("🎤 Accent Classifier App")
st.markdown("Upload an audio file and I'll predict the speaker's accent!")

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

    if suffix == ".mp4":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_file.read())
            tmp_video.flush()
            mp4_path = tmp_video.name

        wav_path = mp4_path.replace(".mp4", ".wav")

        command = [
            "ffmpeg",
            "-y",
            "-i", mp4_path,
            "-ar", "16000",
            "-ac", "1",
            wav_path
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        os.unlink(mp4_path)

        return wav_path
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio = AudioSegment.from_file(uploaded_file)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(tmp_wav.name, format="wav")
            return tmp_wav.name

def predict_accent(file_path):
    try:
        torchaudio.set_audio_backend("sox_io")

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

    except RuntimeError as e:
        if "get_src_stream_info" in str(e):
            time.sleep(1)
            torchaudio.set_audio_backend("sox_io")
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

        raise RuntimeError(f"Prediction error: {e}")

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

        os.remove(wav_path)
