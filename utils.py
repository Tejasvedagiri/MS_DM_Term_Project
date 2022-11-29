import os
from constants import *
import torch
from model import Model

def check_if_dir_exsist():
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

def save_model(model, model_name):
    check_if_dir_exsist()
    torch.save(model, os.path.join(MODEL_DIR, model_name))

def load_model(model, model_name):
    check_if_dir_exsist()
    if not os.path.exists(os.path.join(MODEL_DIR, model_name)):
        return model
    else:
        return torch.load(os.path.join(MODEL_DIR, model_name))