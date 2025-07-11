import torch
from transformers import  AutoModelForSpeechSeq2Seq,AutoProcessor,pipeline
import gradio as gr

device="cuda" if torch.cuda.is_available() else "cpu"
torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32

model_name = "openai/whisper-medium"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name,torch_dtype=torch_dtype,low_cpu_mem_usage=True,use_safetensors=True)
model.to(device)

processor = AutoProcessor.from_pretrained(model_name)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


# 创建Gradio界面
gr.Interface.from_pipeline(pipe).launch()


