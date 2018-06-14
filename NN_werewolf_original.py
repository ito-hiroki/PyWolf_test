# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 21:28:43 2018

@author: kumac
"""

import chainer
import pickle
from chainer.datasets import tuple_dataset
from chainer.datasets import split_dataset_random
from chainer import iterators
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import training
from chainer.training import extensions
import numpy as np

#オリジナルデータ読み込み
with open('gat2017log15_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
        
train = dataset['train']
test = dataset['test']
    
train_data = [[],[]]
test_data = [[],[]]
    
for i in range(len(train)):
    if(train[i][1] == -1):
        train[i][1] = 0
    train_data[0].append(train[i][0])
    train_data[1].append(train[i][1])
print("train:"+str(len(train)))
for i in range(len(test)):
    if(test[i][1] == -1):
        test[i][1] = 0
    test_data[0].append(test[i][0])
    test_data[1].append(test[i][1])
print("test:"+str(len(test)))

train_data[1] = np.array(train_data[1], dtype=np.int32)
test_data[1] = np.array(test_data[1], dtype=np.int32)

train_data = tuple_dataset.TupleDataset(train_data[0], train_data[1])
test_data = tuple_dataset.TupleDataset(test_data[0], test_data[1])

train, valid = split_dataset_random(train_data, int(len(train_data)*0.8), seed=0)

batchsize = 1000

train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)
test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

class MLP(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=2):
        super(MLP, self).__init__()

        # パラメータを持つ層の登録
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        # データを受け取った際のforward計算を書く
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
    
gpu_id = -1
network = MLP()

network = L.Classifier(network)
optimizer = optimizers.Adam().setup(network)
updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

max_epoch = 10

trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='werewolf_result')

trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(valid_iter, network, device=gpu_id), name='val')
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
trainer.extend(extensions.ParameterStatistics(network.predictor.l1, {'std': np.std}))
trainer.extend(extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std.png'))
trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.dump_graph('main/loss'))

trainer.run()