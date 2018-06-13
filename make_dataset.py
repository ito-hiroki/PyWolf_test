# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 01:31:56 2018

@author: kumac
"""

import os
import pandas as pd
import numpy as np
import random
import pickle

train, test = [], []
path = "./result"
for directry in os.listdir(path):
    print("train:"+str(len(train)))
    print("test:"+str(len(test)))
    for name in os.listdir(path + '/' + str(directry)):
        data = pd.read_csv(path + '/' + str(directry) + '/' + str(name))
        print(path + '/' + str(directry) + '/' + str(name))
        valiables = data.columns[1:]
        for i in range(data.shape[0]):
            content = []
            content.append(np.array(data.loc[i][valiables], dtype=np.float32))
            content.append(np.array(data.loc[i]['wolforhuman'], dtype=np.int32))
            if(0 == random.randrange(10)):
                test.append(content)
            else:
                train.append(content)
                
dataset = {"train": train, "test": test}
f = open('gat2017log15_dataset.pickle', 'wb')
pickle.dump(dataset, f)
f.close