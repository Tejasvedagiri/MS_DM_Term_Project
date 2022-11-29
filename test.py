import pandas as pd
from torch.utils.data import DataLoader
from model import Model
import torch
import xgboost as xgb
import numpy as np
import warnings
import os
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

SAMPLE_DATA = 5

torch.manual_seed(42)
df = pd.read_csv("creditcard.csv")
df = df.drop(["Time"], axis=1)#.where(df[df.Class] == True)

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


score_dict["base_line_lgr"] = {"acc_all_data": accuracy_score(y_test, y_pred), "f1_all_data": f1_score(y_test, y_pred, average='macro'),
                                "acc_faulty_data": accuracy_score(faulty_y, y_pred_faulty), "f1_faulty_data": f1_score(faulty_y, y_pred_faulty, average='macro')}

for run in range(0,100):
    failed_index = np.where(y_pred_faulty == 0)
    random_index = np.random.choice(failed_index[0], SAMPLE_DATA)
    noise_data = faulty_transactions_x[random_index]
    
    for model_path in os.listdir("model"):
        model = torch.load(os.path.join("model",model_path)).cpu()

        
        #new_error = model(torch.rand(SAMPLE_DATA, 28).cpu())
        new_error = model(torch.tensor(noise_data).float().cpu())
        new_Y = np.ones((SAMPLE_DATA,1))

        X_train = np.vstack((X_train, new_error[0].detach().numpy()))
        y_train = np.vstack((y_train.reshape(-1,1), new_Y))


        lgr = LogisticRegression(random_state=42)
        lgr.fit(X_train, y_train)
        y_pred = lgr.predict(X_test)
        y_pred_faulty = lgr.predict(faulty_transactions_x)

    score_dict[model_path+"_lgr"] = {"acc_all_data": accuracy_score(y_test, y_pred), "f1_all_data": f1_score(y_test, y_pred, average='macro'),
                                "acc_faulty_data": accuracy_score(faulty_y, y_pred_faulty), "f1_faulty_data": f1_score(faulty_y, y_pred_faulty, average='macro')}

print(score_dict)