import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import uvicorn

# ==============================
# Streaming Whisper 类
# ==============================

class StreamingWhisper:

    def __init__(self):

        print("Loading Whisper model...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_name = "openai/whisper-medium"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )

        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            generate_kwargs={
                "language": "chinese",
                "task": "transcribe",
                "temperature": 0.0
            }
        )

        # ===== 流式核心参数 =====
        self.sample_rate = 16000
        self.max_buffer_seconds = 12     # 保留12秒上下文
        self.decode_interval_seconds = 2 # 每2秒解码一次

        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.last_text = ""
        self.received_samples = 0

        print("Whisper ready.")

    def add_audio(self, new_audio):

        self.audio_buffer = np.concatenate([self.audio_buffer, new_audio])
        self.received_samples += len(new_audio)

        # 保留最近 N 秒
        max_samples = self.sample_rate * self.max_buffer_seconds
        if len(self.audio_buffer) > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]

    def should_decode(self):
        required = self.sample_rate * self.decode_interval_seconds
        return self.received_samples >= required

    def decode(self):

        if len(self.audio_buffer) < self.sample_rate:
            return ""

        self.received_samples = 0

        result = self.pipe(self.audio_buffer)["text"]

        new_text = self.remove_overlap(self.last_text, result)

        self.last_text = result

        return new_text.strip()

    def remove_overlap(self, old, new):

        if not old:
            return new

        if old in new:
            return new.replace(old, "")

        # 更稳健的后缀匹配
        max_overlap = min(len(old), len(new))
        for i in range(max_overlap, 0, -1):
            if old[-i:] == new[:i]:
                return new[i:]

        return new


# ==============================
# FastAPI 服务
# ==============================

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()
    print("Client connected")

    whisper = StreamingWhisper()

    try:
        while True:

            data = await websocket.receive_bytes()

            audio_np = np.frombuffer(data, dtype=np.float32)

            whisper.add_audio(audio_np)

            if whisper.should_decode():

                text = whisper.decode()

                if text != "":
                    await websocket.send_text(text)

    except WebSocketDisconnect:
        print("Client disconnected")


# ==============================
# 启动
# ==============================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
