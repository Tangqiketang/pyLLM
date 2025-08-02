#模型调用
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_dir = "/home/wangmin/model/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3";

model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

outputs = pipe("你好，1+1等于多少",
                    max_length=50, #指定生成文本的最大长度。50表示文本最多包含50个标记（tokens）
                    num_return_sequences=1, #返回多少个独立生成的文本序列。1表示只返回一段文本。
                    truncation=True,#是否截断输入文本以适应模型的最大长度。true时,超出模型最大输入长度的部分将被截断。
                    temperature=0.7,#控制生成的文本的多样性。值越小，生成的文本越保守。
                    top_k=50,#限制模型在每一步生成时仅从概率最高的k个词中选择下一个词。
                    top_p=0.9,#会选择一组累计概率达到p的词汇,一般会高一些
                    clean_up_tokenization_spaces=False #该参数控制生成的文本中是否清理分词时引入的空格。true表示生成的文本会清除多余的空格
                    )
print(outputs)