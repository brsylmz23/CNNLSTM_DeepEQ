# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:54:20 2023

@author: HP
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 18:14:31 2023

@author: brsyl
"""

import torch
import torch.nn.functional as F

#build a block that can be re-used through out the network
#the block contains a skip connection = downsample that is an optional parameter
#note that in the forward pass, skip connnection is applied 
#directly to the inpur of that layer (a^l) and takes it to two layers ahead 
# Örnek bir CNN+LSTM modeli    

#Referenced from the paper of Complex CNN (Ritea, 2018)
class CNNLSTM(torch.nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, output_size):
        super(CNNLSTM, self).__init__()
        self.cnn =  torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # 2D Convolution kullanıyoruz
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 2))
        self.cnn2 =  torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.lstm = torch.nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)


    def forward(self, x,data):
        x = x.to(torch.float32)
        x = self.cnn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cnn2(x)
        
        x = x.permute(0, 2, 3, 1)  # LSTM için boyutları değiştir
        x = x.view(x.size(0), x.size(1), -1)  # LSTM girişi için düzenleme
        x, _ = self.lstm(x)
        out = self.fc(x[:, -1, :])
        return out