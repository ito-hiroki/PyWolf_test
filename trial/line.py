# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:15:16 2018

@author: hiroki
"""

import numpy as np
import sys, os

path = '../aiwolfpy'
current_path = os.path.dirname(os.path.abspath(__name__))
joined_path = os.path.join(current_path, path)
file_path = os.path.normpath(joined_path)
sys.path.append(file_path)
from read_log import read_log

sys.path.append(os.pardir)
from sub_func import name2num

log = read_log('../gat2017log15/000/001.log')

class line(object):
    def __init__(self, num):
        self.player_num = num
        self.relation = np.zeros((self.player_num, self.player_num), dtype = np.int16)
        self.day = 0
        
    def update(self, log):
        answer = []
        for i in range(log.shape[0]):
            if(log['type'][i] == 'initialize'):
                content = log['text'][i].split(" ")
                print(str(content[1])+':'+str(content[2]))
                if(content[2] == 'WEREWOLF'):
                    answer.append('人狼')
                else:
                    answer.append('人間')
                if(len(answer) == self.player_num):
                    print(answer)
                    
            elif(log['type'][i] == 'talk'):
                content = log['text'][i].split(" ")
                # vote宣言
                if(content[0] == 'VOTE'):
                    speak = log['agent'][i]-1
                    to = name2num(content[1])-1
                    self.relation[speak][to] += -1
                # estimate
                elif(content[0] == 'ESTIMATE'):
                    speak = log['agent'][i]-1
                    to = name2num(content[1])-1
                    if(content[2] == 'WEREWOLF'):
                        self.relation[speak][to] += -1
                    else:
                        self.relation[speak][to] += 1
                # request
                elif(content[0][:7] == 'REQUE'):
                    speak = log['agent'][i]-1
                    to = name2num(content[1].strip(')'))-1
                    if(content[0][8:] == 'VOTE'):
                        self.relation[speak][to] += -1
                    elif(content[0][8:] == 'DIVINATION'):
                        self.relation[speak][to] += -1
                # divined
                elif(content[0] == 'DIVINED'):
                    speak = log['agent'][i]-1
                    to = name2num(content[1])-1
                    if(content[2] == 'WEREWOLF'):
                        self.relation[speak][to] += -2
                    else:
                        self.relation[speak][to] += 2
                        
            elif(log['type'][i] == 'vote'):
                speak = log['idx'][i]-1
                to = log['agent'][i]-1
                self.relation[speak][to] += -3
                
            # デバッグようにiday毎に表示
            ahead_day = self.day
            self.day = log['day'][i]
            if(ahead_day != self.day):
                print(self.relation)
                
        return(self.relation)
                    
line_estimate = line(15)
line_estimate.update(log)