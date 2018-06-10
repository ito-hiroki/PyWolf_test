from __future__ import print_function, division 

# this is main script
# simple version test

import aiwolfpy
import aiwolfpy.contentbuilder as cb

import sub_func as sf
import random

myname = 'goldfish'

class SampleAgent(object):
    
    def __init__(self, agent_name):
        # myname
        self.myname = agent_name
        
        
    def getName(self):
        return self.myname
    
    def initialize(self, base_info, diff_data, game_setting):
        self.base_info = base_info
        # game_setting
        self.game_setting = game_setting
        # print(base_info)
        # print(diff_data)
        
        ### 色んな変数定義 ###
        
        # 自分以外のCOを保存 形式:[[**(番号),roll],[**(番号),roll]...]
        self.called_comingout = []
        # 他の人がdivineしたと言った結果を保存 形式:[[**(使用者番号),**(対象者番号),roll],[**(使用者番号),**(対象者番号),roll]...]
        self.called_divined = []
        # 人狼に殺された(死体で発見)人を保存
        self.dead = []
        # vote宣言したかしていないか
        self.vote_declare = 0
        # 霊媒師ローラようにCOされたmediumの数
        self.medium_num = 0
        # 一日の発言回数管理
        self.talk_turn = 0
        
        
    def update(self, base_info, diff_data, request):
        self.base_info = base_info
        #print("base_info:/n")
        #print(base_info)
        #print("diff_data:/n")
        #print(diff_data)
        
        ### 情報集め ###
        
        for i in range(diff_data.shape[0]):
            if(diff_data['type'][i] == 'talk'):
                content = diff_data['text'][i].split(" ")
                
                if(content[0] == 'COMINGOUT'):
                    if(content[2] == 'MEDIUM'):
                        self.medium_num = self.medium_num + 1
                    # [**(番号),roll]
                    self.called_comingout.append([diff_data['agent'][i], content[2]])
                    
                if(content[0] == 'DIVINED'):
                    # [**(使用者番号),**(対象者番号),roll]
                    self.called_divined.append([diff_data['agent'][i], sf.name2num(content[1]), content[2]])
                    
            if(diff_data['type'][i] == 'dead'):
                self.dead.append(diff_data['agent'][i])

               
    def dayStart(self):
        # 初期化
        self.vote_declare = 0
        self.talk_turn = 0
        
        return None
    
    
    def talk(self):
        self.talk_turn += 1
        
        # vote宣言
        if(self.vote_declare != 0):
            self.vote_declare = self.vote()
            return cb.vote(self.vote_declare)
        
        # 発言回数残ってたらskip
        if self.talk_turn <= 10:
            return cb.skip()
        
        return cb.over()
    
    def whisper(self):
        return cb.over()
        
    def vote(self):
        
        # 1. 殺された人or自分を人狼とdivineした占い師が生きてたら無条件で投票
        for i in self.called_divined:
            if(i[2] == 'WEREWOLF' and (i[1] == self.base_info['agentIdx'] or i[1] in self.dead)):
                if(self.base_info['statusMap'][str(i[0])] == 'ALIVE'):
                    idx = i[0]
                    # この場合は一応ESTIMATEした方がよくね？
                    
        # 2. 2日目以降で霊媒師COが2以上の場合ローラー
        if(self.base_info['day'] >= 2 and self.medium_num >= 2):
            for i in self.called_comingout:
                if(i[1] == 'MEDIUM' and self.base_info['statusMap'][str(i[0])] == 'ALIVE'):
                    idx = i[0]
                    
        # 3. わかんねえからランダム投票
        vote_list = []
        for i, status in enumerate(self.base_info['statusMap'].values(), 1):
            if(status == 'ALIVE' and i != self.base_info['agentIdx']):
                vote_list.append(i)
        idx = random.choice(vote_list)
    
        return idx
    
    def attack(self):
        return self.base_info['agentIdx']
    
    def divine(self):
        return self.base_info['agentIdx']
    
    def guard(self):
        return self.base_info['agentIdx']
    
    def finish(self):
        return None
    


agent = SampleAgent(myname)
    


# run
if __name__ == '__main__':
    aiwolfpy.connect_parse(agent)
    