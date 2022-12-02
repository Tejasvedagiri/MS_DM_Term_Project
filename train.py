import torch
from tqdm import tqdm
from torch import nn, optim
from model import Model
from torch.utils.data import DataLoader
from utils import save_model

import pandas as pd
import numpy as np

import argparse

def configure():
    parser = argparse.ArgumentParser()

    parser.add_argument("--H_DIM", type=int, default=32, help="Hidden Dimensions")
    parser.add_argument("--Z_DIM", type=int, default=32, help="Z Dimensions")
    parser.add_argument("--BATCH_SIZE", type=int, default=32, help="Batch size for training")
    parser.add_argument("--NUM_EPOCHS", type=int, default=50, help="To number of epochs to run")
    parser.add_argument("--NO_MODELS", type=int, default=10, help="To models to train")
    parser.add_argument("--LR_RATE", type=int, default=3e-4, help="Uses default 3e-4 Karpathy constant")
    return parser.parse_args()


def train(config):
    dataset = pd.read_csv("creditcard.csv")
    dataset = dataset.drop("Time", axis=1)
    X = dataset.iloc[ : , :-2].to_numpy()
    y = dataset.iloc[ : , -1 ].to_numpy()

    index_y = np.where(y == 1)
    X_fraud = X[index_y]
    y_fraud = y[index_y]

    loader = DataLoader(X_fraud, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)


    for j in range(config.NO_MODELS):
        model = Model(config.INPUT_DIM, config.H_DIM, config.Z_DIM).to(config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config.LR_RATE)
        loss_fn = nn.L1Loss()

        for epoch in range(config.NUM_EPOCHS):
            loop = tqdm(enumerate(loader))
            for i, X in loop:
                X = X.to(config.DEVICE).view(X.shape[0], config.INPUT_DIM).float()
                new_datapoint, mu, sigma = model(X)

                reconstruction_loss = loss_fn(new_datapoint, X)
                kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

                loss = reconstruction_loss + kl_div
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())

        save_model(model, "model_{epoch}.ckpt".format(epoch=j))

if __name__=="__main__":
    config = configure()
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.INPUT_DIM = 28
    train(config)
