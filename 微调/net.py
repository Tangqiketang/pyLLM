from transformers import BertModel
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained = BertModel.from_pretrained('bert-base-chinese',local_files_only=True).to(DEVICE)
##打印模型网络结构
print(pretrained)
print("获取模型网络结构中的参数：",pretrained.embeddings.word_embeddings)

###定义下游任务模型，将主干网络Bert所提取的特征进行fullconnect2分类
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 将上游的768个，变成2分类
        self.fc = torch.nn.Linear(768,2)

    ##bert模型的输入参数
    def forward(self, input_ids, attention_mask, token_type_ids):
        #上游任务不参与训练
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 下游任务参与训练
        out = self.fc(out.last_hidden_state[:, 0, :])  # 取所有批次中第一个token的表示
        out = out.softmax(dim=1)
        return out

