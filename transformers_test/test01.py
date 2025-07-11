##模型下载
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

#输出当前目录路径/mnt/c/MyDataBase/PythonWorkSpace/pyLLM/transformers_test
print(os.getcwd())

model_name = "uer/gpt2-chinese-cluecorpussmall"
cache_dir = "/home/wangmin/model/uer/gpt2-chinese-cluecorpussmall"

#下载模型和分词工具。
AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print("模型分词器已下载到"+cache_dir)