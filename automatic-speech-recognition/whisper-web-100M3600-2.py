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

# 创建支持长音频的pipeline - 优化参数配置
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    chunk_length_s=60,  # 每块60秒
    stride_length_s=[10, 5],  # 增加重叠区域
    generate_kwargs={
        "language": "chinese",  # 明确指定中文
        "task": "transcribe",
        "temperature": 0.0,  # 减少随机性
        "no_repeat_ngram_size": 4,  # 减少重复内容
    }
)


def transcribe_long_audio(audio_path):
    """处理长音频的分块转录"""
    # 使用librosa获取音频长度
    try:
        duration = librosa.get_duration(filename=audio_path)
    except:
        return "无法读取音频文件，请检查格式是否支持（MP3/M4A/WAV）"

    # 根据音频长度动态调整参数
    if duration > 60 * 30:  # 超过30分钟
        chunk_size = 120  # 使用120秒的块
        batch_size = 4  # 减少batch_size防止OOM
    elif duration > 60 * 10:  # 超过10分钟
        chunk_size = 60
        batch_size = 8
    else:
        chunk_size = 30
        batch_size = 16

    # 应用动态参数
    try:
        result = pipe(
            audio_path,
            chunk_length_s=chunk_size,
            stride_length_s=[chunk_size // 6, chunk_size // 10],  # 动态重叠
            batch_size=batch_size
        )["text"]

        # 简单后处理提高可读性
        result = result.replace("。", "。\n")  # 添加分段
        return result
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return "处理失败：显存不足，请尝试上传更短的音频或使用更小的模型"
        return f"处理错误: {str(e)}"
    except Exception as e:
        return f"处理错误: {str(e)}"


# 创建Gradio界面 - 优化界面和说明
interface = gr.Interface(
    fn=transcribe_long_audio,
    inputs=gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="上传音频文件 (MP3/M4A/WAV)",
        max_length=3600  # 最大60分钟录音
    ),
    outputs=gr.Textbox(label="转录结果", lines=20),
    title="高精度会议语音转写系统",
    description=(
        "支持1小时会议录音，最大1GB文件<br>"
        "支持格式: MP3, M4A, WAV<br>"
        "chaos使用Whisper-medium模型提供专业级转录"
    ),
    allow_flagging="never"
)

# 启动应用 - 支持1GB文件
interface.launch(
    max_file_size=1024*1024*1024,  # 1024MB = 1GB
    server_name="0.0.0.0"
)