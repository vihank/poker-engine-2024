import pandas as pd
import numpy as np
from collections import defaultdict

class GameData():
    def __init__(self):
        self.hands = []

    def add_episode(self, info):
        self.hands.append(info)

    def __repr__(self):
        return '\t'.join([x for x in self.hands])

class HandData():
    __allowed = ("isfirst", "reward", "hand", "opp_hand", "turns")
    def __init__(self, **kwarg):
        for k, v in kwarg.items():
            assert(k in self.__class__.__allowed)
            setattr(self, k, v)

    def add_turns(self, turns):
        self.turns = turns

    def add_reward(self, reward):
        self.reward = reward

    def add_num_raise(self, thing):
        self.prop_raise = thing
        
    def __repr__(self):
        return f"IsFirst: {self.isfirst}, Reward: {self.reward}, Hand: {self.hand}, Opponent Hand: {self.opp_hand}, Turns: {self.turns}"


path = "logs/engine_log.csv"
df = pd.read_csv(path)
teamname = "player-bot"
rounds = list()
yourteam = 0
games = GameData()

for i in range(1,1000):
    episode = dict()
    temp = df.loc[df["Round"] == i,]
    fplayer = temp.iloc[0,].loc["Team",]
    if (i == 1 and fplayer == teamname):
        yourteam = 1
    else:
        yourteam = 2
    curr = df.loc[df["Round"] == i,]
    next = df.loc[df["Round"] == i+1].iloc[0,]
    isfirst = (fplayer == teamname)
    reward = next.loc["Bankroll"] - curr.iloc[0,].loc["Bankroll"]

    if (yourteam == 1):
        hand = curr.iloc[0,].loc["Team1Cards"]
        opp_hand = curr.iloc[0,].loc["Team2Cards"]
    else:
        hand = curr.iloc[0,].loc["Team2Cards"]
        opp_hand = curr.iloc[0,].loc["Team1Cards"]

    episode = HandData(isfirst=isfirst, reward=reward, hand=hand, opp_hand=opp_hand)
    actions = []

    for j in range(0,3,1):
        turn = defaultdict(list)
        temp = curr.loc[df["Street"] == j,] 
        if temp.empty:
            break
        turn["street"] = j
        plays = np.shape(temp)[0]
        if (j == 0):
            used = 2
        else:
            used = 0
        left = plays - used
        turn["public_cards"] = temp.iloc[0,].loc["AllCards"]
        while (left > 1):
            betting_round = temp.iloc[used:(used+2),]
            you = betting_round.loc[df["Team"] == teamname,]
            opp = betting_round.loc[-(df["Team"] == teamname),]
            if not you.empty and not opp.empty:
                if "your_action" not in turn:
                    turn["your_action"].append(you.loc[:,"Action"].item())
                    turn["your_amount"].append(you.loc[:,"ActionAmt"].item())
                    turn["opp_action"].append(opp.loc[:,"Action"].item())
                    turn["opp_amount"].append(opp.loc[:,"ActionAmt"].item())
                else:
                    turn["your_action"].append(you.loc[:,"Action"].item())
                    turn["your_amount"].append(you.loc[:,"ActionAmt"].item())
                    turn["opp_action"].append(opp.loc[:,"Action"].item())
                    turn["opp_amount"].append(opp.loc[:,"ActionAmt"].item())
                used += 2
                left = plays - used
        if (left == 1 and (fplayer == teamname)):
            betting_round = temp.iloc[-1,]
            if "your_action" not in turn:
                turn["your_action"].append(you.loc[:, "Action"].item())
                turn["your_amount"].append(you.loc[:, "ActionAmt"].item())
                turn["opp_action"].append("NA")
                turn["opp_amount"].append("NA")

            else:
                turn["your_action"].append(you.loc[:,"Action"].item())
                turn["your_amount"].append(you.loc[:,"ActionAmt"].item())
                turn["opp_action"].append("NA")
                turn["opp_amount"].append("NA")

        actions.append(turn)
        
    episode.add_turns(actions)

    games.add_episode(episode)

# print(games)