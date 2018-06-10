# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 17:27:00 2018

@author: kumac
"""

def name2num(agent_name):
    return int(agent_name[6:8])

def num2name(agent_num):
    return "Agent[" + "{0:02d}".format(agent_num) + "]"

print(num2name(5))
    