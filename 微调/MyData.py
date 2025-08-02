from datasets import load_dataset
from datasets import load_from_disk
from torch.utils.data import Dataset

class Mydataset(Dataset):
    #初始化数据
    def __init__(self,dataType):
        ##下载数据集
        self.dataset = load_dataset("lansinuote/ChnSentiCorp")
        print("数据集所有:",self.dataset)
        if dataType == "train":
            self.dataset = self.dataset["train"]
        elif dataType == "validation":
            self.dataset = self.dataset["validation"]
        elif dataType == "test":
            self.dataset = self.dataset["test"]
        else:
            print("数据集名称错误")
    #获取数据集大小
    def __len__(self):
        return len(self.dataset)
    #对数据做定制化处理
    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]
        return text, label
if __name__ == '__main__':
    dataset = Mydataset("train")
    for data in dataset:
        print(data)