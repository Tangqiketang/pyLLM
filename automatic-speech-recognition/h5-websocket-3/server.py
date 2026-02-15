import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import os


##pip install faster-whisper
##pip install pyannote.audio
##pip install torch torchaudio
##pip install webrtcvad
# ===============================
# 配置
# ===============================

MODEL_SIZE = "large-v2"   # 高精度
DEVICE = "cuda"
COMPUTE_TYPE = "float16"  # 8G显存可跑

HF_TOKEN = "你的huggingface_token"

# ===============================
# 加载模型
# ===============================

print("Loading faster-whisper...")
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

print("Loading speaker diarization...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN
)

print("All models loaded.")

# ===============================
# Streaming 类
# ===============================

class HighAccuracyStreaming:

    def __init__(self):

        self.sample_rate = 16000
        self.max_buffer_seconds = 20
        self.decode_interval = 3

        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.received_samples = 0

        self.last_end_time = 0

    def add_audio(self, audio):

        self.audio_buffer = np.concatenate([self.audio_buffer, audio])
        self.received_samples += len(audio)

        max_samples = self.sample_rate * self.max_buffer_seconds
        if len(self.audio_buffer) > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]

    def should_decode(self):
        return self.received_samples >= self.sample_rate * self.decode_interval

    def decode(self):

        self.received_samples = 0

        segments, info = model.transcribe(
            self.audio_buffer,
            language="zh",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True
        )

        new_text = ""

        for segment in segments:

            if segment.end <= self.last_end_time:
                continue

            self.last_end_time = segment.end
            new_text += segment.text

        return new_text.strip()


# ===============================
# FastAPI
# ===============================

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    stream_engine = HighAccuracyStreaming()

    while True:

        data = await websocket.receive_bytes()
        audio_np = np.frombuffer(data, dtype=np.float32)

        stream_engine.add_audio(audio_np)

        if stream_engine.should_decode():

            text = stream_engine.decode()

            if text:

                # 简体强制化
                text = text.replace("裏", "里")

                await websocket.send_text(text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
