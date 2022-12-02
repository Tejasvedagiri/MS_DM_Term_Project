import pandas as pd
from torch.utils.data import DataLoader
from model import Model
import torch
import xgboost as xgb
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


import argparse
torch.manual_seed(42)

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SAMPLE_DATA", type=int, default=5, help="Sampling size")
    return parser.parse_args()

def plot(df, title, output_filename):
    fig = df.plot(kind="bar").get_figure()
    fig.suptitle(title, fontsize=12)
    plt.xlabel('Models', labelpad=20, rotation=1, fontsize=6)
    plt.ylabel("Scores")
    plt.close(fig)
    fig.savefig("results/"+output_filename)

if __name__=="__main__":
    config = configure()
    df = pd.read_csv("creditcard.csv")
    df = df.drop(["Time"], axis=1)

    faulty_transactions = df.where(df.Class == True).dropna()

    faulty_transactions_x = faulty_transactions.iloc[:, :-2].to_numpy(dtype=float)

    X = df.iloc[:,:-2].to_numpy(dtype=float)
    y = df.iloc[:,-1].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    faulty_y = np.ones(faulty_transactions_x.shape[0])
    score_dict = {}
    lgr = LogisticRegression(random_state=42)
    lgr.fit(X_train, y_train)
    y_pred = lgr.predict(X_test)
    y_pred_faulty = lgr.predict(faulty_transactions_x)


    score_dict["base"] = {"acc_all_data": accuracy_score(y_test, y_pred), "f1_all_data": f1_score(y_test, y_pred, average='macro'),
                                    "acc_faulty_data": accuracy_score(faulty_y, y_pred_faulty), "f1_faulty_data": f1_score(faulty_y, y_pred_faulty, average='macro')}

    for i, model_path in enumerate(os.listdir("model")):
        model = torch.load(os.path.join("model",model_path)).cpu()
        model_name = "m-" + str(i)
        for run in range(0, 1):
            failed_index = np.where(y_pred_faulty == 0)
            random_index = np.random.choice(failed_index[0], config.SAMPLE_DATA)
            noise_data = faulty_transactions_x[random_index]

            new_error = model(torch.tensor(noise_data).float().cpu())
            new_Y = np.ones((config.SAMPLE_DATA,1))

            X_train = np.vstack((X_train, new_error[0].detach().numpy()))
            y_train = np.vstack((y_train.reshape(-1,1), new_Y))


            lgr = LogisticRegression(random_state=42)
            lgr.fit(X_train, y_train)
            y_pred = lgr.predict(X_test)
            y_pred_faulty = lgr.predict(faulty_transactions_x)

        score_dict[model_name] = {"acc_all_data": accuracy_score(y_test, y_pred), "f1_all_data": f1_score(y_test, y_pred, average='macro'),
                                    "acc_faulty_data": accuracy_score(faulty_y, y_pred_faulty), "f1_faulty_data": f1_score(faulty_y, y_pred_faulty, average='macro')}
        
    df2 = pd.DataFrame(score_dict)
    df2.to_csv("results/Results.csv")
    acc_all_data_df = df2.iloc[0]
    f1_all_data = df2.iloc[1]
    acc_faulty_data = df2.iloc[2]
    f1_faulty_data = df2.iloc[3]

    plot(acc_all_data_df, 'Accuracy on complete Dataset', 'acc_all_data_df.png')
    plot(f1_all_data, 'F1 Score on complete Dataset', 'f1_all_data.png')
    plot(acc_faulty_data, 'Accuracy on Only Faulty Transactions', 'acc_faulty_data.png')
    plot(f1_faulty_data, 'F1 on Only Faulty Transactions', 'f1_faulty_data.png')
