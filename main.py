import torch
import torch.nn as nn
from model import Autoencoder
from dataset import MyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用默认的 GPU 设备
else:
    device = torch.device("cpu")
#device = torch.device("cpu")

autoencoder = Autoencoder().to(device)

#开始训练
print("START TRAINING")

# 定义损失函数和优化器
loss_fn = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.00005) #TODO:调参学习率！

num_epochs=300 #TODO:调参epoch！

file_vector = [
    "/work/yeziyi/faster/dataset/Pereira/M02.wq.pkl.dic",
    "/work/yeziyi/faster/dataset/Pereira/M04.wq.pkl.dic",
    "/work/yeziyi/faster/dataset/Pereira/M07.wq.pkl.dic",
    "/work/yeziyi/faster/dataset/Pereira/M15.wq.pkl.dic",
    "/work/yeziyi/faster/dataset/Pereira/P01.wq.pkl.dic"
]

my_dataset = MyDataset(file_vector)

# 使用 DataLoader 加载数据集
dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)

'''
#打印shuffle后的顺序
dataset_size = len(my_dataset)

indices = list(range(dataset_size))

rand_indices = list(dataloader.sampler)

print(f"Indices order : {rand_indices}, len : {len(rand_indices)}")
'''

losses = []

# 训练自编码器
autoencoder.train()
for epoch in range(num_epochs):
    # 前向传播
    for inputs in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
        inputs = inputs.to(device)
        inputs = torch.unsqueeze(inputs, 1)
        current_size = inputs.size(-1)
        padding = (4 - (current_size % 4)) % 4
        inputs = torch.nn.functional.pad(inputs, (0, padding)) #将最后一个维度填充为4的倍数
        outputs = autoencoder(inputs)
        loss = loss_fn(outputs, inputs)
    
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    avg_loss = sum(losses[-len(dataloader):]) / len(dataloader)
    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')


#将cnn编码器处理后的数据存入

dataloader = DataLoader(my_dataset, batch_size=32, shuffle=False)#此时dataloader不能shuffle！血泪教训！！！
count = 0
outputs = []


autoencoder.eval()
with torch.no_grad():
    for inputs in dataloader:
        inputs = inputs.to(device)
        inputs = torch.unsqueeze(inputs, 1)
        inputs = torch.nn.functional.pad(inputs, (0, 3))
        output = autoencoder.cnn_1000(inputs)
        output = output.squeeze()  # 移除维度大小为 1 的维度
        output = output.cpu().numpy()
        for i in range(output.shape[0]):
            outputs.append(output[i])

pcafile_vector = [
    "/work/yeziyi/faster/dataset/Pereira/M02.pca1000.wq.pkl.dic",
    "/work/yeziyi/faster/dataset/Pereira/M04.pca1000.wq.pkl.dic",
    "/work/yeziyi/faster/dataset/Pereira/M07.pca1000.wq.pkl.dic",
    "/work/yeziyi/faster/dataset/Pereira/M15.pca1000.wq.pkl.dic",
    "/work/yeziyi/faster/dataset/Pereira/P01.pca1000.wq.pkl.dic"
]

cnnfile_vector = [
    "/work/wangjiashuo/M02.cnn1000.wq.pkl.dic",
    "/work/wangjiashuo/M04.cnn1000.wq.pkl.dic",
    "/work/wangjiashuo/M07.cnn1000.wq.pkl.dic",
    "/work/wangjiashuo/M15.cnn1000.wq.pkl.dic",
    "/work/wangjiashuo/P01.cnn1000.wq.pkl.dic"
]


for pcafile, cnnfile in zip(pcafile_vector, cnnfile_vector):
    with open(pcafile, 'rb') as file:
        dict = pickle.load(file)

    for key,value in dict.items():
        for i, original_fmri in enumerate(dict[key]['fmri']):
            updated_fmri = outputs[count]
            count+=1
            dict[key]['fmri'][i] = updated_fmri
    with open(cnnfile, 'w+b') as file:
        file.truncate(0)
        pickle.dump(dict, file)
    print("count:",count)
print("DONE")