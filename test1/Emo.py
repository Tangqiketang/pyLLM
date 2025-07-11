from transformers import AutoTokenizer
### 处理数据集
import torch
import requests

print('torch版本：' + torch.__version__)
print('cuda是否可用：' + str(torch.cuda.is_available()))
print('cuda版本：' + str(torch.version.cuda))
print('cuda数量:' + str(torch.cuda.device_count()))
print('GPU名称：' + str(torch.cuda.get_device_name()))


API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"
response = requests.post(API_URL,json={"inputs":"你好啊，huggingface"})
print(response.json())

