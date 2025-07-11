from transformers import BertTokenizer,BertForSequenceClassification
from transformers import pipeline

##加载模型和分词器
model_name = "bert-base-chinese"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

#创建分类
classifier = pipeline("text-classification", model=model,tokenizer=tokenizer)
result = classifier("我今天很开心")
print(result)