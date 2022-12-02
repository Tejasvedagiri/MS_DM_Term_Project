# Variational Autoencoders to Generate synthetic dataset


# Python set-up Cond GPU
```
conda create -n DM_VAE python=3.8 -y 
conda activate DM_VAE
conda install pandas scikit-learn tqdm xgboost matplotlib seaborn -c conda-forge -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y 
```

# Training the model
We train the VAE model to generate 10 new models that is used 
```
python train.py
```

## Optional Arguments to fine tune model
```
optional arguments:
  -h, --help            show this help message and exit
  --H_DIM H_DIM         Hidden Dimensions
  --Z_DIM Z_DIM         Z Dimensions
  --BATCH_SIZE BATCH_SIZE
                        Batch size for training
  --NUM_EPOCHS NUM_EPOCHS
                        To number of epochs to run
  --NO_MODELS NO_MODELS
                        To models to train
  --LR_RATE LR_RATE     Uses default 3e-4 Karpathy constant
```

# Evaulating the model and generating results
```
```
