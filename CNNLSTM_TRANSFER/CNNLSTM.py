
import torch
import torch.nn as nn

# Örnek bir CNN modeli
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
device = torch.device(dev)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.sonv = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(7, 1), stride=(2, 2), padding=(1, 1))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(4,4), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 2))
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3), stride=(3,3), padding=(1, 1))
    
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.sonv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        
        x = x.permute(0, 2, 3, 1)  # LSTM için boyutları değiştir
        return x

# Örnek bir LSTM modeli
class LSTM(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)  # LSTM girişi için düzenleme
        
        batch_size = x.size(0)
        
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        x, _ = self.lstm(x,(hidden,cell))
        # Diğer LSTM katmanları ve işlemleri buraya eklenebilir
        x = self.fc(x[:, -1, :])  # Son zaman adımı için lineer katmana geçiş
        x = self.relu(x)
        return x

# Ana model
class CNNLSTM(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, output_size):
        super(CNNLSTM, self).__init__()
        self.cnn = CNN()
        self.lstm = LSTM(input_channels, hidden_size, num_layers, output_size)

    def forward(self, x,data,data_window):
        splits = torch.split(x, split_size_or_sections=data_window, dim=2)  
        cnn_outputs = [self.cnn(split) for split in splits]  # Her bir parçayı CNN'ye besler

        # Her bir CNN çıktısını ayrı ayrı LSTM'ye besler
        lstm_outputs = [self.lstm(output) for output in cnn_outputs]

        # LSTM çıktılarını birleştirir
        combined_output = torch.cat(lstm_outputs, dim=1)
        output = lstm_outputs[-1]

        return output