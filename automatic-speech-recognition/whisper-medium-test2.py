import librosa
from transformers import pipeline
import torch
import gc
# 用 Librosa 加载音频
speech_file = "./20241023135746-CG周会-纯音频-1.m4a"
waveform, sample_rate = librosa.load(speech_file, sr=16000, mono=True)  # 转换为模型所需的格式

# 更激进的流式处理设置
pipe = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-medium",  # 使用更小的模型
    chunk_length_s=15,  # 更小的块大小
    stride_length_s=[4, 1],  # 更小的步长
    device="cuda" if torch.cuda.is_available() else "cpu"
)


# 使用生成器实现真正的流式输出
def realtime_stream(waveform, sample_rate, pipe, chunk_size=5):
    total_samples = len(waveform)
    chunk_samples = chunk_size * sample_rate

    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[start:end]

        output = pipe({"raw": chunk, "sampling_rate": sample_rate})
        yield output['text'].strip()

        start = end


# 使用方式
for i, result in enumerate(realtime_stream(waveform, sample_rate, pipe)):
    if result:
        print(f"片段 {i + 1}: {result}")

# 推理结束后添加
torch.cuda.empty_cache()
gc.collect()