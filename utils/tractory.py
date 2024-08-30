folder_path = 'data\\csv'

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=100, output_size=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1] 

file_list = []
for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    file_list.append(item_path)
    
data_list = []
for csv in file_list:
    df = pd.read_csv(csv).dropna()
    coordinate = df[['Lat', 'Lon']]
    data_list.append(coordinate)

for data_all in data_list:
    data_all = (data_all - data_all.mean()) / data_all.std()

    data_all = torch.tensor(data_all[['Lon', 'Lat']].values, dtype=torch.float32)

    seq_length = 10  
    sequences = [data_all[i:i+seq_length] for i in range(len(data_all)-seq_length)]
    X = torch.stack([seq[:-1] for seq in sequences]) 
    Y = torch.stack([seq[-1] for seq in sequences])  

    model = LSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1000):
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
        
        optimizer.zero_grad()
        
        loss = 0
        for i in range(len(X)):
            output = model(X[i])
            loss += criterion(output, Y[i])
        loss = loss / len(X)  
        
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    outputs = [model(seq[:-1]).detach().numpy() for seq in X]
    outputs = np.array(outputs)
    plt.scatter(data_all.numpy()[:, 0], data_all.numpy()[:, 1], color='b', label='Original data', s=3)
    plt.scatter(outputs[:, 0], outputs[:, 1], color='r', label='Fitted data')
    plt.plot(outputs[:, 0], outputs[:, 1], color='r', label='Fitted line')

    plt.legend()
    plt.show()


print('over')
