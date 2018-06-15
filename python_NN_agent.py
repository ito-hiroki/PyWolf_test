from __future__ import print_function, division 

# this is main script
# simple version test

import aiwolfpy
import aiwolfpy.contentbuilder as cb

import sub_func as sf
import numpy as np
from chainer import serializers
import chainer 
import chainer.links as L
import chainer.functions as F
from chainer.datasets import tuple_dataset
import random



myname = 'goldfish'

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

class SampleAgent(object):
    
    def __init__(self, agent_name):
        # myname
        self.myname = agent_name
        
        self.infer_net = MLP()
        serializers.load_npz('./results/werewolf_result_batchsize14/snapshot_epoch-2', \
                             self.infer_net, path='updater/model:main/predictor/')
        
        
        
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
        
        # 学習用のデータ
        self.info_list = []
        # その日に取得した学習用のデータ
        self.info = np.zeros((15,11), dtype=np.float32)
        # 最初に一回記録する用のフラグ
        self.onlyone = True
        # CO 占い師数
        self.seer_co_num = 0
        # CO 霊媒師数
        self.medium_co_num = 0
        # 占い師COした順番
        self.seer_co_order = 0
        # 霊媒師COした順番
        self.medium_co_order = 0
        # vote宣言の内容
        self.infoTalkvote = np.zeros((15,2))
        
        
        
    def update(self, base_info, diff_data, request):
        self.base_info = base_info
        #print("base_info:/n")
        #print(base_info)
        print("diff_data:")
        print(diff_data)
        
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
                
        ### NN用のデータを集める ###
        # 最初に記録する情報
        if(self.onlyone):
            for i in range(len(self.base_info['statusMap'])):
                # 生きてるか死んでいるか(死:1,生:0)
                if(self.base_info['statusMap'][str(i+1)] == 'DEAD'):
                    self.info[i][0] = 1
                # 日にちを入れる
                self.info[i][10] = self.base_info['day']
            self.onlyone = False
        
        # diff_dataの中をなめていく
        for i in range(diff_data.shape[0]):
            # 始めにtypeがtalkだった場合(今回はほとんどこれ)
            if(diff_data['type'][i] == 'talk'):
                content = diff_data['text'][i].split(" ")
                # 内容がCOの場合
                if(content[0] == 'COMINGOUT' and diff_data['agent'][i] == sf.name2num(content[1])):
                    # 霊媒師かつ発話エージェントと発話対象が同じであれば
                    if(content[2] == 'MEDIUM'):
                        self.medium_co_num += 1
                        self.medium_co_order += 1
                        self.info[diff_data['agent'][i]-1][2] = self.medium_co_order
                    # 霊媒師かつ発話エージェントと発話対象が同じであれば
                    if(content[2] == 'SEER'):
                        self.seer_co_num += 1
                        self.seer_co_order += 1
                        self.info[diff_data['agent'][i]-1][1] = self.seer_co_order
                # 内容がDIVINED
                if(content[0] == 'DIVINED'):
                    # 人狼判定
                    if(content[2] == 'WEREWOLF'):
                        # 受けた人狼判定数更新
                        self.info[sf.name2num(content[1])-1][4] += 1
                        # 出した人狼判定数更新
                        self.info[diff_data['agent'][i]-1][8] += 1
                    if(content[2] == 'HUMAN'):
                        # 受けた人間判定数更新
                        self.info[sf.name2num(content[1])-1][3] += 1
                        # 出した人狼判定数更新
                        self.info[diff_data['agent'][i]-1][7] += 1
                #内容がVOTE
                if(content[0] == 'VOTE'):
                    # 投票先
                    self.infoTalkvote[diff_data['agent'][i]-1][0] = sf.name2num(content[1])
                    # 日にち
                    self.infoTalkvote[diff_data['agent'][i]-1][1] = self.base_info['day']
            # typeがvoteだった場合
            if(diff_data['type'][i] == 'vote'):
                # 同じ日のtalkで話した投票先とvote先が異なるときに1を代入
                if(self.infoTalkvote[diff_data['idx'][i]-1][0] != diff_data['agent'][i] \
                   and self.infoTalkvote[diff_data['idx'][i]-1][1] == diff_data['day'][i]):
                    self.info[diff_data['idx'][i]-1][9] = 1
                    
        # 会話が1ターン終わったらco占い師数とco霊媒師数を代入する
        for i in range(len(self.base_info['statusMap'])):
            # co占い師数
            self.info[i][1] = self.seer_co_num
            # co霊媒師数
            self.info[i][2] = self.medium_co_num

    def dayStart(self):
        # 初期化前に保存しとくもの
        print(self.info)
        self.info_list.append(np.array((self.info), dtype=np.float32))
        """
        x = self.info[14, ...]
        print(x)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = self.infer_net(x)
        print('予測ラベル:', y.argmax(axis=1)[0])
        """
        
        # 初期化
        self.vote_declare = 0
        self.talk_turn = 0
        
        self.onlyone = True
        return None
    
    
    def talk(self):
        self.talk_turn += 1
        
        # vote宣言
        if(self.vote_declare != self.vote()):
            self.vote_declare = self.vote()
            print(self.vote_declare)
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
                    return idx
                    # この場合は一応ESTIMATEした方がよくね？
                    
        # 2. 2日目以降で霊媒師COが2以上の場合ローラー
        if(self.base_info['day'] >= 2 and self.medium_num >= 2):
            for i in self.called_comingout:
                if(i[1] == 'MEDIUM' and self.base_info['statusMap'][str(i[0])] == 'ALIVE'):
                    idx = i[0]
                    return idx
                    
        # 3. NN使って投票
        #x = tuple(np.array(self.info))
        x = self.info
        print(x)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = self.infer_net(x)
        y = y.array
        result = y.argmax(axis=1)
        for i, num in enumerate(result):
            if(self.base_info['statusMap'][str(i+1)] == 'ALIVE' and i == 1):
                idx = num+1
                print(idx)
                return idx
                
        # 4. わかんねえからランダム投票
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
    