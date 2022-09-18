#!/usr/bin/env python

import os, warnings, sys
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.utils as utils
from algorithm.model.classifier import Classifier, get_data_based_model_params
from algorithm.utils import get_model_config



# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(data, data_schema, hyper_params):  
    
    # set random seeds
    utils.set_seeds()
    
    # balance the target classes  
    # data = utils.get_resampled_data(data = data, 
    #                     max_resample = model_cfg["max_resample_of_minority_classes"])
    # print(data.head()); sys.exit()      
    
    # perform train/valid split 
    train_data, valid_data = train_test_split(data, test_size=model_cfg['valid_split'])
    # print(train_data.shape, valid_data.shape) #; sys.exit()    

    # preprocess data
    print("Pre-processing data...")
    train_X, train_y, valid_X, valid_y , preprocessor = preprocess_data(train_data, valid_data, data_schema)
    # print("train/valid data shape: ", train_X.shape, train_y.shape, valid_X.shape, valid_y.shape)
    
    # balance the targetclasses  
    train_X, train_y = get_resampled_data(train_X, train_y)
    valid_X, valid_y = get_resampled_data(valid_X, valid_y)
    
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


def preprocess_data(train_data, valid_data, data_schema):        
    
    # print('Preprocessing train_data of shape...', train_data.shape)
    pp_params = pp_utils.get_preprocess_params(data_schema)    
    
    preprocess_pipe = pp_pipe.get_preprocess_pipeline(pp_params, model_cfg)
    train_data = preprocess_pipe.fit_transform(train_data)
    train_X, train_y = train_data['X'].astype(np.int32), train_data['y'].astype(np.int32)
    
    if valid_data is not None:
        valid_data = preprocess_pipe.transform(valid_data)
        valid_X, valid_y = valid_data['X'].astype(np.int32), valid_data['y'].astype(np.int32)    

    return train_X, train_y, valid_X, valid_y, preprocess_pipe



def get_resampled_data(X, y):    
    # if some minority class is observed only 1 time, and a majority class is observed 100 times
    # we dont over-sample the minority class 100 times. We have a limit of how many times
    # we sample. max_resample is that parameter - it represents max number of full population
    # resamples of the minority class. For this example, if max_resample is 3, then, we will only
    # repeat the minority class 2 times over (plus original 1 time). 
    max_resample = model_cfg["max_resample_of_minority_classes"]
    
    # class_count = list(y.sum(axis=0))
    class_counts = np.asarray(np.unique(y, return_counts=True)).T
    max_obs_count = max(class_counts[:, 1])
    
    resampled_X, resampled_y = [], []
    for i, count in list(class_counts):
        count = int(count)
        if count == 0: continue
        # find total num_samples to use for this class
        size = max_obs_count if max_obs_count / count < max_resample else count * max_resample
        size = int(size)
        # print(i, count, size)
        # if observed class is 50 samples, and we need 125 samples for this class, 
        # then we take the original samples 2 times (equalling 100 samples), and then randomly draw
        # the other 25 samples from among the 50 samples
        full_samples = size // count
        idx = y == i
        for _ in range(full_samples):
            resampled_X.append(X[idx, :])
            resampled_y.append(y[idx])
            
        # find the remaining samples to draw randomly
        remaining =  int(size - count * full_samples   )
        sampled_idx = np.random.randint(count, size=remaining)
        resampled_X.append(X[idx, :][sampled_idx, :])
        resampled_y.append(y[idx][sampled_idx])
        
    resampled_X = np.concatenate(resampled_X, axis=0)
    resampled_y = np.concatenate(resampled_y, axis=0)
    # print(resampled_X.shape, resampled_y.shape)
    # shuffle the arrays
    resampled_X, resampled_y = shuffle(resampled_X, resampled_y)
    
    return resampled_X, resampled_y