import torch
from tqdm import tqdm
from torch import nn, optim
from model import Model
from torch.utils.data import DataLoader
from utils import save_model

import pandas as pd
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 28
H_DIM = 32
Z_DIM = 32
NUM_EPOCHS = 50
BATCH_SIZE = 32
LR_RATE = 3e-4  # Karpathy constant

dataset = pd.read_csv("creditcard.csv")
dataset = dataset.drop("Time", axis=1)
X = dataset.iloc[ : , :-2].to_numpy()
y = dataset.iloc[ : , -1 ].to_numpy()


index_y = np.where(y == 1)
X_fraud = X[index_y]
y_fraud = y[index_y]

loader = DataLoader(X_fraud, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

create_models = 20
for j in range(create_models):
    model = Model(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
    loss_fn = nn.L1Loss()

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(loader))
        for i, X in loop:
            X = X.to(DEVICE).view(X.shape[0], INPUT_DIM).float()
            new_datapoint, mu, sigma = model(X)

            reconstruction_loss = loss_fn(new_datapoint, X)
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    save_model(model, "model_{epoch}.ckpt".format(epoch=j))