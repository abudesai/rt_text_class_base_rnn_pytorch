#!/usr/bin/env python

import os, warnings, sys
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.utils as utils
from algorithm.model.classifier import Classifier, get_data_based_model_params
from algorithm.utils import get_model_config
import torch.optim as optim
from torch.nn import GRU, LSTM, MaxPool1d, ReLU, Linear, Embedding, Module, CrossEntropyLoss
import torch




# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(data, data_schema, hyper_params):  
    
    # set random seeds
    utils.set_seeds()
    
    # balance the target classes  
    data = utils.get_resampled_data(data = data, 
                        max_resample = model_cfg["max_resample_of_minority_classes"])
    # print(data.head()); sys.exit()      
    
    # perform train/valid split 
    train_data, valid_data = train_test_split(data, test_size=model_cfg['valid_split'])
    # print(train_data.shape, valid_data.shape) #; sys.exit()    

    # preprocess data
    print("Pre-processing data...")
    train_X, train_y, valid_X, valid_y , preprocessor = preprocess_data(train_data, valid_data)
    # print("train/valid data shape: ", train_X.shape, train_y.shape, valid_X.shape, valid_y.shape)
    
    # Create and train model   
    model, history = train_model(train_X, train_y, valid_X, valid_y, hyper_params)    
    
    return preprocessor, model, history


def train_model(train_X, train_y, valid_X, valid_y, hyper_params):    
    # get model hyper-parameters parameters 
    
    data_based_params = get_data_based_model_params(train_X, train_y, valid_X, valid_y)
    model_params = { **data_based_params, **hyper_params }
    # print(model_params) #; sys.exit()
    
    # Create and train model   
    model = Classifier(  **model_params )  
    # model.summary()  ; sys.exit()
      
    print('Fitting model ...')  
    history = model.fit(
        train_X=train_X, train_y=train_y, 
        valid_X=valid_X, valid_y=valid_y,
        epochs = 1000,  # we have early stopping, so this is max epochs actually
        verbose = 1, 
    )      
    return model, history


def preprocess_data(train_data, valid_data):    
    
    preprocess_pipe = pp_pipe.get_preprocess_pipeline(model_cfg)
    train_data = preprocess_pipe.fit_transform(train_data)
    train_X, train_y = train_data['X'].astype(np.int32), train_data['y'].astype(np.int32)
    
    if valid_data is not None:
        valid_data = preprocess_pipe.transform(valid_data)
        valid_X, valid_X = valid_data['X'].astype(np.int32), valid_data['y'].astype(np.int32)    

    return train_X, train_y, valid_X, valid_X, preprocess_pipe

