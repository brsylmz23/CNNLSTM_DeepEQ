# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:54:33 2024

@authors: Barış Yılmaz, Melek Türkmen, Erdem Akagündüz
"""

from __future__ import print_function
import os
import torch
from torch.utils.data import DataLoader
from evaluate import testModel
from functions import datasetCreator, structureData, getArguments, plot_

#filter warnings
import warnings
warnings.filterwarnings("ignore")
import pickle

##############################################################################   
def test(modelFolder,testDataFolder):
    
    # check for GPU, use if there is one
    if (torch.cuda.is_available()):
        seed_value = 42
        torch.cuda.manual_seed_all(seed_value)
        print(torch.cuda.get_device_name(0))
    else:   
        print("no GPU found")        
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    
    # create arguments set. 
    # this is necessary for training, 
    # for this testing code most parameters are redundant, 
    # bc we use a pretrained model
    kwargs = getArguments(testDataFolder)
    
    
    # load the training statistics dictionary (input mean etc.)
    # these are to be used during inference
    # file_path = os.path.join(os.getcwd(),modelFolder,'training_data_stats.pkl')    
    # with open(file_path, 'rb') as file: 
    #     trainStats = pickle.load(file)
        
    # load the dataset from the testDataFolder    
    testData = datasetCreator(**kwargs)
    test_set = structureData(testData, phase = "test", **kwargs)    
    # parameters for model evaluation and the loader object
    params = {'batch_size': kwargs.get('batchsize'), 'shuffle': False}   
    # the loader object of the test set
    testLoader = DataLoader(test_set, **params)
                     
    # load the pretrained model
    model_path = os.path.join(os.getcwd(),modelFolder,'finalModel.pt')
    model = torch.load(model_path)
 
    # evaluate the tet set with the model
    pred, gt =  testModel(testLoader,testData, model, **kwargs)  
    
    return pred
    
testDataFolder = r"sampleQuakes"
modelFolder = r"model" 
pred = test(modelFolder,testDataFolder)