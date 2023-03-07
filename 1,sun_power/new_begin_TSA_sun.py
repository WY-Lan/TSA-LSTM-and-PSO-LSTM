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
    data = np.array(pd.read_excel('testing.xlsx')).astype('float32')
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



def TSA_LSTM(iw, maxrun, N, D, ST, maxFEs, dmin, dmax):
    low = int(N * 0.1)
    high = int(N * 0.25)

    for run in range(maxrun):
        np.random.seed()
        # 重复的次数
        trees = np.zeros((N, D))
        obj_trees = np.zeros((N, 1))  # 存放每个个体的目标值

        for i in range(N):
            for j in range(D):
                if j == 1:
                    trees[i, j] = dmin[j] + np.random.rand() * (dmax[j] - dmin[j])
                else:
                    trees[i, j] = int(dmin[j] + np.random.rand() * (dmax[j] - dmin[j]))
            obj_trees[i] = train_LSTM(trees[i, :])
        FEs = N  # 因为当前已经产生了N个个体

        minimum = np.min(obj_trees)
        iter1 = 0

        while (FEs <= maxFEs):
            iter1 = iter1 + 1
            for i in range(N):
                # i是树木
                ns = int(low + (high - low) * np.random.rand()) + 1
                FEs = FEs + ns
                if ns > high:
                    ns = high

                seeds = np.zeros((ns, D))  # 记录当前的种子的具体形式
                obj_seeds = np.zeros((ns, 1))
                minimum, min_index = np.min(obj_trees), np.argmin(obj_trees)
                bestParams = trees[min_index, :]

                for j in range(ns):
                    # j 是种子
                    komus = int(np.random.rand() * N) + 1
                    while komus == i or komus >=N or komus < 0 :
                        komus = int(np.random.rand() * N) + 1
                    seeds[j, :] = trees[j, :]

                    for d in range(D):
                        if np.random.rand() < ST:
                            if d == 1:
                                seeds[j, d] = iw * trees[i, d] + (bestParams[d] - trees[komus, d]) * (np.random.rand() - 0.5) * 2
                            else:
                                seeds[j, d] = int(iw * trees[i, d] + (bestParams[d] - trees[komus, d]) * (np.random.rand() - 0.5) * 2)
                            if seeds[j, d] > dmax[d]:
                                seeds[j, d] = dmax[d]
                            if seeds[j, d] < dmin[d]:
                                seeds[j, d] = dmin[d]
                        else:
                            if d == 1:
                                seeds[j, d] = iw * trees[i, d] + (trees[i, d] - trees[komus, d]) * (np.random.rand() - 0.5) * 2
                            else:
                                seeds[j, d] = int(iw * trees[i, d] + (trees[i, d] - trees[komus, d]) * (np.random.rand() - 0.5) * 2)
                            if seeds[j, d] > dmax[d]:
                                seeds[j, d] = dmax[d]
                            if seeds[j, d] < dmin[d]:
                                seeds[j, d] = dmin[d]
                    obj_seeds[j] = train_LSTM(seeds[j, :])

                mini_seeds, mini_seeds_ind = np.min(obj_seeds), np.argmin(obj_seeds)

                if mini_seeds < obj_trees[i]:
                    trees[i, :] = seeds[mini_seeds_ind, :]
                    obj_trees[i] = mini_seeds

            min_tree, min_tree_index = np.min(obj_trees), np.argmin(obj_trees)
            if min_tree < minimum:
                minimum = min_tree
                bestParams = trees[min_tree_index, :]

            print('Iter={} .... min={} .... FES={} .... \n'.format(iter1, minimum, FEs))
            with open('TSA_result.csv','a+',newline='') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(np.append(bestParams,minimum))#用于以后画收敛曲线
        print('Run={} .... min={} ....\n'.format(run, minimum))

    return bestParams,minimum

if __name__ == '__main__':
    iw = 1
    maxrun = 1
    N = 20
    D = 4
    ST = 0.1
    maxFEs = 50
    dmin = [500, 0.00001, 5, 1]
    dmax = [2000, 0.1, 30, 5]
    TSA_bestParams, TSA_minimum = TSA_LSTM(iw, maxrun, N, D, ST, maxFEs, dmin, dmax)
    print(TSA_bestParams, TSA_minimum)
