import torch
from torch.utils.data import DataLoader, Dataset
import pickle

# 自定义一个数据集类，继承自 torch.utils.data.Dataset
class MyDataset(Dataset):
    def __init__(self, file_vector):
        self.data_fmri = []
        for file_name in file_vector:
            self.load_file(file_name)

    def load_file(self,file_name):
        with open(file_name, 'rb') as file:
            data_dict = pickle.load(file)
        self.data_dict = data_dict
        for key in self.data_dict.keys():
            for data in self.data_dict[key]['fmri']:
                #TODO:是否需要去0？暂时没有去0
                data_tensor = torch.Tensor(data.reshape(88, 128, 85)) 
                #data_tensor = torch.Tensor(data)
                self.data_fmri.append(data_tensor)

    def __len__(self):
        return len(self.data_fmri)

    def __getitem__(self, index):
        return self.data_fmri[index]