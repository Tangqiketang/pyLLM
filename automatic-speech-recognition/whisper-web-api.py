import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import shutil
import os

### http传递音频整体文本输出 Content-Type：multipart/form-data
## file:aa.mp

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

app = FastAPI()

# 创建处理长音频的转录函数
def transcribe_long_audio(audio_path):
    """处理长音频的分块转录"""
    try:
        duration = librosa.get_duration(filename=audio_path)
    except:
        raise HTTPException(status_code=400, detail="无法读取音频文件，请检查格式是否支持（MP3/M4A/WAV）")

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
            raise HTTPException(status_code=500, detail="处理失败：显存不足，请尝试上传更短的音频或使用更小的模型")
        raise HTTPException(status_code=500, detail=f"处理错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理错误: {str(e)}")


# 定义接收音频文件的路由
@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # 保存上传的音频文件
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(file.file, temp_file)

    # 处理音频并获取转录结果
    try:
        transcription = transcribe_long_audio(temp_file_path)
        return {"transcription": transcription}
    except HTTPException as e:
        return e.detail
    finally:
        # 删除临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# 启动 FastAPI 服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
