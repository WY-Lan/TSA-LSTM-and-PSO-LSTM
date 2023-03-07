import numpy as np
import pandas as pd

df = np.array(pd.read_csv("real.csv"))[:,-1]
df_PSO = np.array(pd.read_csv("PSO_pred.csv"))[:,-1]
df_TSA = np.array(pd.read_csv('TSA_pred.csv'))[:,-1]
df_regular = np.array(pd.read_csv('regular_pred.csv'))[:,-1]

def MSE(y,yhat):
    return np.sum((y - yhat)**2)/len(y)

def RMSE(y, yhat):
    return np.sqrt(MSE(y, yhat))

def MAPE(y, yhat):
    return np.sum(np.abs((y+1e-12 - yhat))/(y+1e-12))/len(y)

def MAE(y, yhat):
    return np.sum(np.abs(y - yhat))/len(y)

res = np.zeros((4,3))
for i in range(4):
    for j in range(3):
        if i == 0:
            if j == 0:
                res[i][j] = MSE(df,df_regular)
            elif j == 1:
                res[i][j] = MSE(df,df_PSO)
            elif j == 2:
                res[i][j] = MSE(df, df_TSA)
        elif i == 1:
            if j == 0:
                res[i][j] = RMSE(df,df_regular)
            elif j == 1:
                res[i][j] = RMSE(df,df_PSO)
            elif j == 2:
                res[i][j] = RMSE(df, df_TSA)
        elif i == 2:
            if j == 0:
                res[i][j] = MAE(df,df_regular)
            elif j == 1:
                res[i][j] = MAE(df,df_PSO)
            elif j == 2:
                res[i][j] = MAE(df, df_TSA)
        elif i == 3:
            if j == 0:
                res[i][j] = MAPE(df,df_regular)
            elif j == 1:
                res[i][j] = MAPE(df,df_PSO)
            elif j == 2:
                res[i][j] = MAPE(df, df_TSA)

df = pd.DataFrame(res)
df.to_csv('alllll_result.csv')