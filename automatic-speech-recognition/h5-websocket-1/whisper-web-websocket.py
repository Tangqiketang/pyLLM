import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import uvicorn

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_name = "openai/whisper-medium"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)

model.to(device)

processor = AutoProcessor.from_pretrained(model_name)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={
        "language": "chinese",
        "task": "transcribe",
        "temperature": 0.0
    }
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()
    print("Client connected")

    try:
        while True:

            data = await websocket.receive_bytes()

            audio_np = np.frombuffer(data, dtype=np.float32)

            # Whisper 需要 float32 numpy 16000Hz
            result = pipe(audio_np)["text"]

            await websocket.send_text(result)

    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
