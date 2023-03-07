''' 目前已知参数是多少，只需要调整相应的参数，这里主要做的更改就是将把输出改成了指定的形式，便于后续画图和计算各项指标 '''
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
import math
import csv
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'



# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x


if __name__ == '__main__':

    # checking if GPU is available
    device = torch.device("cpu")

    if (torch.cuda.is_available()):
        device = torch.device("cuda:0")
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
    1896.8003600599313, 0.0197719249657055, 24.593179318527724, 3.632221396562308, 0.043090518563985825
    # 数据读取&类型转换
    epoch = int(1896)
    lr = 0.0197719249657055
    cell_size = int(24.593179318527724)
    num_layer = int(3.632221396562308)
    name = 'PSO'

    data = np.array(pd.read_excel('AirQualityUCI.xlsx')).astype('float32')
    data = data[~np.isnan(data).any(axis=1), :]

    #归一化
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data_x = data[:,0:-1]
    data_y = data[:,-1]

    # 数据集分割
    data_len = len(data_x)
    t = np.linspace(0, data_len, data_len + 1)

    train_data_ratio = 0.8  # Choose 80% of the data for training
    train_data_len = int(data_len * train_data_ratio)

    train_x = data_x[5:train_data_len]
    train_y = data_y[5:train_data_len]
    t_for_training = t[5:train_data_len]

    test_x = data_x[train_data_len:]
    test_y = data_y[train_data_len:]
    t_for_testing = t[train_data_len:]

    # ----------------- train -------------------
    INPUT_FEATURES_NUM = train_x.shape[1]
    OUTPUT_FEATURES_NUM = 1
    train_x_tensor = train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  # set batch size to 1
    train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 1

    # transfer data to pytorch tensor
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)

    lstm_model = LstmRNN(INPUT_FEATURES_NUM, cell_size, output_size=OUTPUT_FEATURES_NUM, num_layers=num_layer)  # 20 hidden units
    lstm_model.to(device)
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)
    print('train x tensor dimension:', Variable(train_x_tensor).size())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)
    prev_loss = 1000
    max_epochs = epoch

    train_x_tensor = train_x_tensor.to(device)
    train_y_tensor = train_y_tensor.to(device)

    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor).to(device)
        loss = criterion(output, train_y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < prev_loss:
            torch.save(lstm_model.state_dict(), 'lstm_model.pt')  # save model parameters to files
            prev_loss = loss

        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        else:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    # prediction on training dataset
    pred_y_for_train = lstm_model(train_x_tensor).to(torch.device("cpu"))
    pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # ----------------- test -------------------
    lstm_model = lstm_model.eval()  # switch to testing model

    # prediction on test dataset
    test_x_tensor = test_x.reshape(-1, 1, INPUT_FEATURES_NUM)
    test_x_tensor = torch.from_numpy(test_x_tensor)  # 变为tensor
    test_x_tensor = test_x_tensor.to(device)

    pred_y_for_test = lstm_model(test_x_tensor).to(torch.device("cpu"))
    pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y))
    print("test loss：", loss.item())

    # ----------------- plot -------------------
    '''plt.figure()
    plt.plot(t_for_training, train_y, 'b', label='y_trn')
    plt.plot(t_for_training, pred_y_for_train, 'y--', label='pre_trn')'''


    ''' 这里注意修改，然后注意修改的时TSA或者是PSO，或者是普通的LSTM '''
    df = pd.DataFrame(test_y)
    df.to_csv(name+'_test.csv')
    df = pd.DataFrame(pred_y_for_test)
    df.to_csv(name+'_pred.csv')
