from transformers import BertTokenizer


token = BertTokenizer.from_pretrained("bert-base-chinese")
print(token)

vocab = token.get_vocab()
print(vocab)

token.add_tokens("阳光")
token.add_special_tokens({"eos_token":"[EOS]"})
vocab = token.get_vocab()

print("阳光" in vocab, "大地" in vocab, "[EOS]" in vocab)

##编码新句子
out = token.encode(text="阳光照在大地上[EOS]",
                   text_pair=None,
                   truncation=True,
                   padding="max_length",
                   max_length=10,
                   add_special_tokens=True,
                   return_tensors=None)
print("out:",out)
print("out.after.decode:",token.decode(out))


