# Variational Autoencoders to Generate synthetic dataset


# Python set-up Cond GPU
```
conda create -n DM_VAE python=3.8 -y 
conda activate DM_VAE
conda install pandas scikit-learn tqdm xgboost matplotlib seaborn -c conda-forge -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y 
```

# Python set-up conda cpu
```
conda create -n DM_VAE python=3.8 -y 
conda activate DM_VAE
conda install pandas scikit-learn tqdm xgboost matplotlib seaborn -c conda-forge -y
conda install pytorch torchvision torchaudio cpuonly -c pytorch
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
python test.py
```
## Optional Arguments
```
optional arguments:
  -h, --help            show this help message and exit
  --SAMPLE_DATA SAMPLE_DATA
                        Sampling size
```

# Results
The following are the results generated from the model

## All data accuracy
We the base line model has an accuracy og 99.9% so as all the other models. So we insted check the accuracy of the model on only fraudulent dataset.
<img src=results/acc_all_data_df.png>

## Fraudulent dataset Accuracy
While testing the base line we get an approximate score of 63% and while running on various models we were able to get a max of 75% on selected models and on most models we were able to generate an accuracy of 70%

<img src=results/acc_faulty_data.png>

## F-1 Score on all Data
<img src=results/f1_all_data.png>

## F-1 Score on Fraudulent Dataset
<img src=results/f1_faulty_data.png>
