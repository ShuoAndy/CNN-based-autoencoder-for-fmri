import torch
import torch.nn as nn
import torch.nn.init as init

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.SiLU(),
            nn.Conv3d(1, 4, kernel_size=3, padding=1),  
            nn.MaxPool3d(kernel_size=2, stride=2),  # 最大池化
            nn.SiLU(),
            nn.Conv3d(4, 16, kernel_size=3, padding=1),  
            nn.MaxPool3d(kernel_size=2, stride=2),  # 最大池化
            nn.SiLU(),
            nn.Conv3d(16, 64, kernel_size=3, padding=1),  
            nn.MaxPool3d(kernel_size=2, stride=2)  # 最大池化
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 转置卷积层
            nn.SiLU(),
            nn.ConvTranspose3d(16, 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # 转置卷积层
            nn.SiLU(),
            nn.ConvTranspose3d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 转置卷积层
            nn.SiLU(),
        )
        self.fc_layer_1 = nn.Linear(64 * 11 * 16 * 11, 1000)
        self.fc_layer_2 = nn.Linear(1000 , 64 * 11 * 16 * 11)


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init.xavier_uniform_(m.weight)  # 对卷积层权重进行Xavier初始化

    def cnn_1000(self,x):
        x = self.encoder(x)

        x = x.view(x.size(0), -1)
        x = self.fc_layer_1(x)
        return x

    def forward(self, x):
        batch_len = len(x)
        x = self.encoder(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc_layer_1(x)

        #print(x.shape)
        x = self.fc_layer_2(x)
        x = x.view(batch_len, 64, 11, 16, 11)

        x = self.decoder(x)
        #x = self.tanh(x)
        #print(x.shape)

        return x

