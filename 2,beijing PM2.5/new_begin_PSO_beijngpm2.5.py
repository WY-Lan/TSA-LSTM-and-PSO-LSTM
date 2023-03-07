import scipy.io as sio
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import math
import csv
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


def train_LSTM(X):

    device = torch.device("cuda:0")

    epoch = int(X[0])
    lr = X[1]
    hidden_size = int(X[2])
    num_layers = int(X[3])

    # 数据读取&类型转换
    data = np.array(pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')).astype('float32')
    data = data[~np.isnan(data).any(axis=1), :]

    #归一化
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data_x = data[:, 0:-1]
    data_y = data[:, -1]

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

    lstm_model = LstmRNN(INPUT_FEATURES_NUM, hidden_size, output_size=OUTPUT_FEATURES_NUM,
                         num_layers=num_layers)  # 20 hidden units
    lstm_model.to(device)
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)
    print('train x tensor dimension:', Variable(train_x_tensor).size())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)

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

        '''if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.10f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        else:
            print('Epoch: [{}/{}], Loss:{:.10f}'.format(epoch + 1, max_epochs, loss.item()))'''

    # prediction on training dataset
    pred_y_for_train = lstm_model(train_x_tensor).to(torch.device("cpu"))
    pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # ----------------- test -------------------
    lstm_model = lstm_model.eval()  # switch to testing model

    # prediction on test dataset
    test_x_tensor = test_x.reshape(-1, 1,
                                   INPUT_FEATURES_NUM)
    test_x_tensor = torch.from_numpy(test_x_tensor)  # 变为tensor
    test_x_tensor = test_x_tensor.to(device)

    pred_y_for_test = lstm_model(test_x_tensor).to(torch.device("cpu"))
    pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y))

    print(X)
    print("test loss：", loss.item())

    return loss.item()


def PSO_LSTM(N, D, dmin, dmax, maxiter):
    c1 = 2
    c2 = 2
    w = 0.5
    pN = N  # 粒子数量
    dim = D  # 搜索维度

    DOWN = dmin
    UP = dmax

    X = np.zeros((pN, dim))  # 所有粒子的位置和速度
    V = np.zeros((pN, dim))
    pbest = np.zeros((pN, dim))  # 个体经历的最佳位置和全局最佳位置
    gbest = np.zeros(dim)
    p_fit = np.zeros(pN)  # 每个个体的历史最佳适应值

    fit = 1
    for i_episode in range(maxiter):
        """初始化s"""
        np.random.seed()
        # 初始粒子适应度计算
        print("计算初始全局最优")
        for i in range(pN):
            for j in range(dim):
                V[i][j] = np.random.random()
                if j == 1:
                    X[i][j] = DOWN[j] + (UP[j] - DOWN[j])*np.random.random()
                else:
                    X[i][j] = int(DOWN[j] + (UP[j] - DOWN[j])*np.random.random())
            pbest[i] = X[i]  # 个人历史最优

            p_fit[i] = train_LSTM(X[i])
            if p_fit[i] < fit:
                fit = p_fit[i]
                gbest = X[i]

        for j in range(maxiter):

            for i in range(pN):
                temp = train_LSTM(X[i])
                with open('PSO_result.csv' ,'a+' ,newline='') as f:
                    csv_write = csv.writer(f)
                    csv_write.writerow(np.append(gbest,fit))  # 用于以后画收敛曲线
                if temp < p_fit[i]:  # 更新个体最优
                    p_fit[i] = temp
                    pbest[i] = X[i]
                    if p_fit[i] < fit:  # 更新全局最优
                        gbest = X[i]
                        fit = p_fit[i]

            # 更新位置
            for i in range(pN):
                # 这里先不用管个体的数量是整数还是小数，先不用考虑
                V[i] = w * V[i] + c1 * np.random.random() * (pbest[i] - X[i]) + c2 * np.random.random() *(gbest - X[i])
                ww = 1
                for k in range(dim):
                    if DOWN[k] < X[i][k] + V[i][k] < UP[k]:
                        continue
                    else:
                        ww = 0
                X[i] = X[i] + V[i] * ww
    return gbest, fit

if __name__ == '__main__':
    N = 10
    D = 4
    dmin = [500, 0.00001, 5, 1]
    dmax = [2000, 0.1, 30, 5]
    maxiter = 70
    PSO_bestParams, PSO_minimum = PSO_LSTM(N, D, dmin, dmax, maxiter)
    print(PSO_bestParams, PSO_minimum)
