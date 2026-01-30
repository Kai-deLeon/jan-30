import torch
import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")

X = torch.tensor(data.drop("Y", axis=1).to_numpy()).float()

Y = torch.tensor(data["Y"].to_numpy()).reshape(-1, 1)

w = torch.tensor([
    [-2.0],
    [1.0]
])

b = torch.tensor([
    [-2.0]
])

Yhat = X@w + b

r = Yhat - Y

SSE = r.T@r
loss = SSE/2

print(loss)