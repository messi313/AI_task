import streamlit as st
import joblib
import torch
import numpy as np
from speechbrain.pretrained import EncoderClassifier
import tempfile
import os
from pydub import AudioSegment
from pydub.utils import which
import subprocess

# Configure ffmpeg/ffprobe for pydub
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

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
