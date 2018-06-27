from __future__ import print_function, division 

import aiwolfpy
import aiwolfpy.contentbuilder as cb

import sub_func as sf
import numpy as np
import random

# chainer用(アカンかったらここ消せ)
import chainer_predict_kai
from chainer_predict_kai import MLP
from chainer import serializers
import os


class Goldfish(object):
    
    def __init__(self, agent_name):
        # myname
        self.myname = agent_name
        self.infer_net = MLP()
        serializers.load_npz(os.path.dirname(__file__)+'/snapshot_epoch-8', \
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
        # bodyguardの時は使ってる！
        self.called_comingout = []
        # 他の人がdivineしたと言った結果を保存 形式:[[**(使用者番号),**(対象者番号),roll],[**(使用者番号),**(対象者番号),roll]...]
        self.called_divined = []
        # 人狼に殺された(死体で発見)人を保存
        self.dead = []
        # vote宣言したかしていないか
        self.vote_declare = 0
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
        # 霊媒結果の保存  形式:[[**(番号), roll], [**(番号), roll]...]
        self.result_roll = []
        # 結果を発表したかしてないかFlag
        self.not_reported = True
        # CO用
        self.comingout = ''
        # 霊媒師のCO日を決める
        self.comingoutday = random.choice([2,3])
        # 裏切者が即COor対抗COっぽく見せかける？and白だしor黒出し?
        self.possess_justco = random.choice([0,1])
        # 裏切者が黒出ししたやつを記録
        self.possess_wolf = []
        # 占い師(狂人)が投票した奴を記録
        self.voted_list = []
        # wisper結果の保存
        self.result_wisper = []
        # 村人数
        self.menber_num = 0
        # 5任用
        self.num5_div = 0
        # 5人人狼でseerを殺せたっぽいとき
        self.like_possess = []
        
        
    def update(self, base_info, diff_data, request):
        self.base_info = base_info
        self.menber_num = len(self.base_info['statusMap'])
        #print("base_info:/n")
        #print(base_info)
        #print("diff_data:")
        #print(diff_data)
        
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
            content = diff_data['text'][i].split(" ")
            if(diff_data['type'][i] == 'talk'):
                # 内容がCOの場合
                if(content[0] == 'COMINGOUT' and diff_data['agent'][i] == sf.name2num(content[1])):
                    # 一応誰が何をCOしたのか記録
                    # [**(番号),roll]
                    self.called_comingout.append([diff_data['agent'][i], content[2]])
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
                    # [**(使用者番号),**(対象者番号),roll]
                    self.called_divined.append([diff_data['agent'][i], sf.name2num(content[1]), content[2]])
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

            # typeがidentifyだった場合
            if(diff_data['type'][i] == 'identify'):
                # 結果を公開したかしてないか
                self.not_reported = True
                # 霊媒結果の保存  形式:[[**(番号), roll], [**(番号), roll]...]
                self.result_roll.append([diff_data['agent'][i], content[2]])
                
            # typeがdivineだった場合
            if(diff_data['type'][i] == 'divine'):
                # 結果を公開したかしてないか
                self.not_reported = True
                # 霊媒結果の保存  形式:[[**(番号), roll], [**(番号), roll]...]
                self.result_roll.append([diff_data['agent'][i], content[2]])
                
            # typeがwisperだった場合
            if(diff_data['type'][i] == 'wisper'):
                # ささやき結果の保存  形式:[[**(番号), roll], [**(番号), roll]...]
                self.result_wisper.append([diff_data['agent'][i], content[2]])
            
            # typeがdeadだった場合(白確の保存)
            if(diff_data['type'][i] == 'dead'):
                self.dead.append(diff_data['agent'][i])

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
        
        # 初期化
        self.vote_declare = 0
        self.talk_turn = 0
        self.possess_justco = random.choice([0,1])
        
        self.onlyone = True
        return None
    
    
    def talk(self):
        self.talk_turn += 1
        # 5任用
        if(self.menber_num == 5):
            if(self.base_info['myRole'] == 'VILLAGER' and self.base_info['myRole'] == 'WEREWOLF'):
                if(self.talk_turn < 3):
                    return cb.skip()
                return cb.vote(self.vote())
            if(self.base_info['myRole'] == 'SEER'):
                if(self.comingout == ''):
                    self.comingout = 'SEER'
                    return cb.comingout(self.base_info['agentIdx'], self.comingout)
                if(self.comingout != '' and self.not_reported):
                    divination = self.result_roll.pop()
                    self.not_reported = False
                    if(divination[1] == 'WEREWOLF'):
                        self.num5_div = divination[0]
                        return cb.divined(divination[0], divination[1])
                    else:
                        for i in range(1, self.menber_num+1):
                            if(self.base_info['statusMap'][str(i)] == 'ALIVE'):
                                self.num5_div = i
                                return cb.divined(i, 'WEREWOLF')
            if(self.base_info['myRole'] == 'POSSESS'):
                if(self.comingout == ''):
                    self.comingout = 'SEER'
                    return cb.comingout(self.base_info['agentIdx'], self.comingout)
                if(self.comingout != '' and self.not_reported):
                    self.not_reported = False
                    possess_div = []
                    for i in self.called_comingout:
                        if(i[1] != 'SEER'):
                            possess_div.append(i[0])
                    self.num5_div = random.choice(possess_div)
                    return cb.divined(self.num5_div, 'WEREWOLF')
                if(self.base_info['day'] == 2):
                    self.comingout = 'POSSESS'
                    return cb.cb.comingout(self.base_info['agentIdx'], self.comingout)
            # 4.発言回数残ってたらskip
            if self.talk_turn <= 10:
                return cb.skip()
            return cb.over()
                                
        
        # 1.CO
        if(self.base_info['myRole'] == 'MEDIUM' and self.comingout == '' \
            and ( self.seer_co_num >= 2 or self.comingoutday <= self.base_info['day'])):
            self.comingout = 'MEDIUM'
            return cb.comingout(self.base_info['agentIdx'], self.comingout)
        elif(self.base_info['myRole'] == 'SEER' and self.comingout == ''):
            self.comingout = 'SEER'
            return cb.comingout(self.base_info['agentIdx'], self.comingout)
        elif(self.base_info['myRole'] == 'POSSESS' and self.comingout == '' \
             and (self.possess_justco == 1 or self.seer_co_num >= 1)):
            self.comingout = 'SEER'
            return cb.comingout(self.base_info['agentIdx'], self.comingout)
        
        # 2. 結果報告
        if(self.not_reported):
            if(self.base_info['myRole'] == 'MEDIUM' and self.result_roll != []):
                ident = self.result_roll.pop()
                return cb.identified(ident[0], ident[1])
            elif(self.base_info['myRole'] == 'SEER' and self.result_roll != []):
                divination = self.result_roll.pop()
                return cb.divined(divination[0], divination[1])
            if(self.result_roll == []):
                self.not_reported = False
        if(self.base_info['myRole'] == 'POSSESS' and self.comingout == 'SEER'):
            # self.possess_justcoこれ使ってるけどco関係なくてただの欄数が欲しいだけ
            idx = chainer_predict_kai.estimate_wolf(self.info, self.base_info, self.infer_net)
            if(self.possess_justco == 1 and idx != -1):
                possess_div = []
                for i in range(1,len(self.base_info['statusMap'])+1):
                    if((i not in idx) and (i not in self.voted_list)):
                        possess_div.append(i)
                div = random.choice(possess_div)
                self.voted_list.append(div)
                species = random.choice(['HUMAN','WEREWOLF'])
                if(species == 'WEREWOLF'):
                    self.possess_wolf.append(div)
                return cb.divination(div, species)
            else:
                not_vote_list = []
                for i in idx:
                    if(i not in self.voted_list):
                        not_vote_list.append(i)
                voted = random.choice(not_vote_list)
                self.voted_list.append(voted)
                return cb.divination(voted, 'HUMAN')
                
        
        # 3.vote宣言
        # 狂人は黒判定した奴に投票
        if(self.base_info['myRole'] == 'POSSESS' and self.possess_wolf != []):
            return cb.vote(random.choice(self.possess_wolf))
        elif(self.vote_declare != self.vote()):
            self.vote_declare = self.vote()
            print("self.vote_declare")
            print(self.vote_declare)
            return cb.vote(self.vote_declare)
        
        # 4.発言回数残ってたらskip
        if self.talk_turn <= 10:
            return cb.skip()
        
        return cb.over()
    
    def whisper(self):
        
        if(self.base_info['day'] == 0):
            # 最初にwisperで何騙りするか決定
            choice_roll = random.choice(range(10))
            if(choice_roll < 3):
                self.comingout = 'SEER'
                return cb.comingout(self.base_info['agentIdx'], self.comingout)
            elif(choice_roll < 4):
                self.comingout = 'MEDIUM'
                return cb.comingout(self.base_info['agentIdx'], self.comingout)
            else:
                self.comingout = 'VILLAGER'
                return cb.comingout(self.base_info['agentIdx'], self.comingout) 
                
        if(self.base_info['day'] == 0 and self.result_wisper != []):
            for i in self.result_wisper:
                if(i[1] == 'WEREWOLF'):
                    self.comingout = 'VILLAGER'
                    return cb.comingout(self.base_info['agentIdx'], self.comingout)
        
        return cb.over()
        
    def vote(self):
        # 5人人狼の場合
        if(self.menber_num == 5):
            not_vote_list = []
            vote_list = []
            if(self.base_info['myRole'] == 'SEER'):
                return self.num5_div
            if(self.base_info['myRole'] == 'POSSESS'):
                return self.num5_div
            if(self.base_info['myRole'] == 'WEREWOLF'):
                if(self.base_info['day'] == 1 and self.seer_co_num < 3):
                    for i in self.called_divined:
                        if(self.base_info['statusMap'][str(i[0])] == 'ALIVE'):
                            not_vote_list.append(i[0])
                            not_vote_list.append(i[1])
                    for i in range(1,self.menber_num+1):
                        if(i not in not_vote_list and self.base_info['statusMap'][str(i)] == 'ALIVE'):
                            vote_list.append(i)
                    return random.choice(vote_list)
                else:
                    for i in range(1, self.menber_num+1):
                        if(self.base_info['statusMap'][str(i)] == 'ALIVE' and i not in self.like_possess and i != self.base_info['agentIdx']):
                            vote_list.append(i)
                    return random.choice(vote_list)
                    
            if(self.base_info['myRole'] == 'VILLAGER'):
                for i in self.called_divined:
                    if(i[1] == self.base_info['agentIdx'] and i[2] == 'WEREWOLF'):
                        return i[0]
                    elif(i[1] != self.base_info['agentIdx'] and i[2] == 'WEREWOLF'):
                        vote_list.append(i[1])
                    not_vote_list.append(i[0])
                if(vote_list == []):
                    for i in range(1, 6):
                        if(i not in not_vote_list):
                            vote_list.append(i)
                return random.choice(vote_list)
                
                     
            
        
        # 霊媒師の場合、自分黒判定or偽霊媒師が現れた時は投票
        if(self.base_info['myRole'] == 'MEDIUM'):
            for i in self.called_divined:
                if(i[1] == self.base_info['agentIdx'] and i[2] == 'WEREWOLF'):
                    idx = i[0]
                    return idx
            for i in self.called_comingout:
                if(i[1] == 'MEDIUM'):
                    idx = i[0]
                    return idx
                    
        # 裏切者場合、seerに投票
        if(self.base_info['myRole'] == 'POSSESS'):
            for i in self.called_comingout:
                if(i[1] == 'MEDIUM'):
                    idx = i[0]
                    return idx
                    
        # 1. 殺された人or自分を人狼とdivineした占い師が生きてたら無条件で投票
        for i in self.called_divined:
            if(i[2] == 'WEREWOLF' and (i[1] == self.base_info['agentIdx'] or i[1] in self.dead)):
                if(self.base_info['statusMap'][str(i[0])] == 'ALIVE'):
                    idx = i[0]
                    return idx
        # 2. 2日目以降で霊媒師COが2以上の場合ローラー
        if(self.base_info['day'] >= 2 and self.medium_co_num >= 2):
            for i in self.called_comingout:
                if(i[1] == 'MEDIUM' and self.base_info['statusMap'][str(i[0])] == 'ALIVE'):
                    idx = i[0]
                    return idx
                    
        # 3. NN使って投票
        #x = tuple(np.array(self.info))
        idx = chainer_predict_kai.estimate_wolf(self.info, self.base_info, self.infer_net)
        if(idx != -1):
            return random.choice(idx)
                
        # 4. わかんねえからランダム投票
        vote_list = []
        for i, status in enumerate(self.base_info['statusMap'].values(), 1):
            if(status == 'ALIVE' and i != self.base_info['agentIdx']):
                vote_list.append(i)
        idx = random.choice(vote_list)
    
        return idx
    
    def attack(self):
        if(self.menber_num == 5):
            not_attack_list = []
            attack_list = []
            seer_killed = False
            for i in self.called_divined:
                if(self.base_info['statusMap'][i[0]] == 'ALIVE'):
                    if(i[1] == self.base_info['agentIdx'] and i[2] == 'WEREWOLF'):
                        seer_killed = True
                        return i[0]
                    if(seer_killed):
                        self.like_possess.append(i[0])
                    not_attack_list.append(i[0])
                    not_attack_list.append(i[1])
            for i in range(1, self.menber_num+1):
                if(i not in not_attack_list and self.base_info['statusMap'][str(i)] == 'ALIVE'):
                    attack_list.append(i)
            return random.choice(attack_list)
        
        idx = chainer_predict_kai.estimate_wolf(self.info, self.base_info, self.infer_net)
        if(idx != -1):
            wolf_attack = []
            for i in range(1,len(self.base_info['statusMap'])+1):
                if((i not in idx) and (i not in self.voted_list)):
                    wolf_attack.append(i)
            if(wolf_attack != []):
                return random.choice(wolf_attack)
            
        # 4. わかんねえからランダムattack
        attack_list = []
        for i, status in enumerate(self.base_info['statusMap'].values(), 1):
            if(status == 'ALIVE' and i != self.base_info['agentIdx'] and (str(i) not in self.base_info['roleMap'].keys())):
                attack_list.append(i)
        idx = random.choice(attack_list)
        
        return idx
    
    def divine(self):
        # 5任用
        if(self.menber_num == 5):
            divine_list = []
            for i in range(1, self.menber_num+1):
                if(self.base_info['statusMap'][str(i)] == 'ALIVE'):
                    divine_list.append(i)
            idx = random.choice(divine_list)
            self.voted_list.append(idx)
            return idx
        
        idx = chainer_predict_kai.estimate_wolf(self.info, self.base_info, self.infer_net)
        if(idx != -1):
            return random.choice(idx)
        
        vote_list = []
        for i, status in enumerate(self.base_info['statusMap'].values(), 1):
            if(status == 'ALIVE' and i != self.base_info['agentIdx'] and (i not in self.voted_list)):
                vote_list.append(i)
        idx = random.choice(vote_list)
        self.voted_list.append(idx)
        return idx
    
    def guard(self):
        
        # 狩人
        if(self.base_info['myRole'] == 'BODYGUARD'):
            # 占い師2以上、霊媒師1以上 であれば 霊媒def
            if(self.seer_co_num >= 2 and self.medium_co_num == 1):
                for i in self.called_comingout:
                    if(i[1] == 'MEDIUM'):
                        return i[0]
            # 占い師2以上、霊媒師1以上 であれば 霊媒def
            if(self.seer_co_num >= 2 and self.medium_co_num > 2):
                seer_list = []
                for i in self.called_comingout:
                    if(i[1] == 'SEER'):
                        seer_list.append(i[0])
                return random.choice(seer_list)
            # 占い師
            if(self.seer_co_num == 1):
                for i in self.called_comingout:
                    if(i[1] == 'SEER'):
                        return i[0]
            # わかんねえからランダムdef
            def_list = []
            for i, status in enumerate(self.base_info['statusMap'].values(), 1):
                if(status == 'ALIVE' and i != self.base_info['agentIdx']):
                    def_list.append(i)
            return random.choice(def_list)
    
    def finish(self):
        return None
    

myname = "Goldfish"
agent = Goldfish(myname)
    


# run
if __name__ == '__main__':
    aiwolfpy.connect_parse(agent)
    