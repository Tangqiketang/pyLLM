import torch
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer
from torch.optim import AdamW

from 微调.MyData import Mydataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10

#token = BertTokenizer.from_pretrained("bert-base-chinese",cache_dir="./cache")
token = BertTokenizer.from_pretrained("bert-base-chinese",local_files_only=True)


#自定义函数对数据进行编码。训练时再编码，不然工作量很大
def collate_fn(batch):
    sentes = [i[0] for i in batch]
    label = [i[1] for i in batch]
    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sentes,
                                    truncation=True,padding=True,
                                    max_length=512,return_tensors="pt",
                                    return_length=True
                                    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)
    return input_ids, attention_mask, token_type_ids, labels

#创建数据集
train_dataset = Mydataset("train")
#创建Dataloader
train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=collate_fn)

if __name__ == '__main__':
    ##开始训练
    print(DEVICE)
    model = Model().to(DEVICE)
    optimizer = AdamW(model.parameters(),lr=5e-4)
    loss_fnc = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        for i,(input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            #将数据放到device上
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            ##执行前向计算得到输出
            out = model(input_ids, attention_mask, token_type_ids)

            loss = loss_fnc(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%5 ==0:
                out = out.argmax(dim=1)
                acc = (out == labels).sum().item()/len(labels)
                print(epoch,i,loss.item(),acc)
        ##保存模型参数
        torch.save(model.state_dict(),f"/home/wangmin/modelTestParam/{epoch}bert.pt")
        print(epoch,"参数保存成功")