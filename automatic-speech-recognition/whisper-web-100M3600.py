import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gradio as gr
import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 加载模型和处理器
model_name = "openai/whisper-medium"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_name)

# 创建支持长音频的pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    # 关键：启用长音频分块处理
    chunk_length_s=30,  # 每块30秒
    stride_length_s=[6, 4],  # 块间重叠
)


def transcribe_long_audio(audio_path):
    """处理长音频的分块转录"""
    # 使用librosa获取音频长度
    duration = librosa.get_duration(filename=audio_path)

    # 根据音频长度动态调整参数
    if duration > 60 * 10:  # 超过10分钟
        chunk_size = 60  # 使用60秒的块
    else:
        chunk_size = 30  # 默认30秒块

    # 应用动态参数
    return pipe(
        audio_path,
        chunk_length_s=chunk_size,
        stride_length_s=[chunk_size / 5, chunk_size / 7],  # 动态重叠
        batch_size=8  # 根据GPU内存调整
    )["text"]


# 创建Gradio界面
interface = gr.Interface(
    fn=transcribe_long_audio,
    inputs=gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="上传音频文件",
        # 关键：限制文件大小（100MB）
        max_length=3600  # 最大60分钟录音
    ),
    outputs=gr.Textbox(label="转录结果"),
    title="语音转写系统",
    description="上传音频文件，最大支持1小时录音"
)

# 启动应用（添加文件大小限制）
interface.launch(
    max_file_size=100*1024*1024,  # 限制上传文件为100MB
)