from trun import Game
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
import pandas as pd
import numpy as np
from collections import defaultdict
from python_skeleton.skeleton.evaluate import evaluate

class GameData():
    def __init__(self):
        self.rounds = []

    def map_cards(self):
        pass

    def cur_in(self, player):
      if player:
        return self.player_in
      return self.player_opp

    def frequent_card_count(self, hand):
      ranks = [int(card[0]) for card in hand]
      counts = [ranks.count(x) for x in ranks]
      return max(counts)

    def max_straight_count(self, hand):
      
      ranks = [int(card[0]) for card in hand]
      ranks.sort()
      max_streak = 0
      cur = ranks[0] - 1
      cur_streak = 0
      for i in range(len(ranks)):
        if ranks[i] == cur + 1:
          cur_streak += 1
        else:
          cur_streak = 1
        max_streak = max(max_streak, cur_streak)
        cur = ranks[i]

      return max_streak

    def frequent_suit_count(self, hand):
      ranks = [card[1] for card in hand]
      counts = [ranks.count(x) for x in ranks]
      return max(counts)

    def max_card(self, hand):
        s = [int(h[0]) for h in hand]
        return max(s)

    def extract_hand(self, hand):
        if hand[-1] == " ":
          hand = hand[:-1]
        handL = hand.split()
        
        return (self.max_card(handL), self.frequent_card_count(handL), self.frequent_suit_count(handL), self.max_straight_count(handL))

    def map_action(self, action, amount, player):
        if action == "fold":
          return 0
        elif action == "check":
          return 1
        elif action == "call":
          return 2
        if player:
          self.player_in += amount
        else:
          if amount != "NA":
            self.player_opp += amount
          else:
            return 0
        if amount < 5:
          return 3
        elif amount < 25:
          return 4
        elif amount < 125:
          return 5
        return 6

    def format_turn(self, episode, turn, isfirst, bankroll, hand):
        if isinstance(turn["public_cards"], float):
            turn["public_cards"] = ""
        episode["a"].append(self.map_action(turn["your_action"], turn["your_amount"], True))
        opponent_cost = self.map_action(turn["opp_action"], turn["opp_amount"], False)
        hand_s = hand + " " + "".join(turn["public_cards"])
        episode["s"].append((turn["street"], isfirst, bankroll, self.cur_in(True), self.cur_in(False), *self.extract_hand(hand_s), evaluate(hand_s.split()[:2], hand_s.split()[2:]) / 1000 ))
        episode["r"].append(0)

    def add_episode(self, info):
        self.player_in = 2 - info.isfirst
        self.player_opp = 1 + info.isfirst
        episode = {"s": [], "a": [], "r": []}
        for turn in info.turns:
          self.format_turn(episode, turn, info.isfirst, info.bankroll, info.hand)
        episode["r"][-1] = info.reward
        episode["s"].append(None)
        self.rounds.append(episode)

    def __repr__(self):
        return '\t'.join([x for x in self.hands])

class HandData():
    __allowed = ("isfirst", "reward", "hand", "opp_hand", "turns", "bankroll")
    def __init__(self, **kwarg):
        for k, v in kwarg.items():
            assert(k in self.__class__.__allowed)
            setattr(self, k, v)

    def add_turns(self, turns):
        self.turns = turns


game = Game()

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

ACTIONS = ["Fold", "Check", "Call", "Raise0", "Raise1", "Raise2", "Raise3",]
OBSERVATIONS = ["STREET", "BANKROLL", "OUR_IN", "THEIR_IN", "MAX_CARD", "COMMON_RANK", "COMMON_SUIT", "MAX_STRAIGHT", "EQUITY"]
# Get number of actions from gym action space
n_actions = len(ACTIONS)
# Get the number of state observations
n_observations = len(OBSERVATIONS)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

action_probs = torch.ones(n_observations) # TODO: Fill in

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[torch.multinomial(action_probs, 1)]], device=device, dtype=torch.long)


episode_durations = []

def optimize_model():
    print("Optimizing")
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

# state is init state
# info is ??

for i_episode in range(num_episodes):
    # print(games.rounds[i_episode]["s"])
    state_round = game.run_round(False)
    state = torch.tensor(state_round, dtype=torch.float32, device=device).unsqueeze(0)
    format_state = state
    for t in count():
        action = select_action(format_state)
        new_state = game.run_turn(action)
        observation, reward = game.rounds[i_episode]["s"][t + 1], games.rounds[i_episode]["r"][t] # TODO: Fill in
        reward = torch.tensor([reward], device=device)

        if observation is None:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

    print(reward)