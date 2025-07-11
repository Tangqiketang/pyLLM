import librosa
from transformers import pipeline
import torch
import gc
# 用 Librosa 加载音频
speech_file = "/mnt/c/MyDataBase/PythonWorkSpace/pyLLM/automatic-speech-recognition/a.mp3"
waveform, sample_rate = librosa.load(speech_file, sr=16000, mono=True)  # 转换为模型所需的格式
# 创建管道
pipe = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-medium",
    chunk_length_s=30  # 处理长音频分段
)
# 传递音频数据
result = pipe({"raw": waveform, "sampling_rate": sample_rate})
print(result)

# 推理结束后添加
torch.cuda.empty_cache()
gc.collect()