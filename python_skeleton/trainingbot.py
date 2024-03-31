"""
Simple example pokerbot, written in Python.
"""

import itertools
import random
import pickle
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from skeleton.actions import Action, CallAction, CheckAction, FoldAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.evaluate import evaluate

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.activation = nn.ReLU(n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.activation(self.layer3(x))
class GameData():
    def __init__(self, street, pips, hand):
        self.state = (street, 400 - pips[0], 400 - pips[1], *self.extract_hand(hand), evaluate(hand[0], hand[1]) / 1000 )

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
        hand = hand[0] + hand[1]

        return (self.max_card(hand), self.frequent_card_count(hand), self.frequent_suit_count(hand), self.max_straight_count(hand))

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

class TrainingPlayer(Bot):
    """
    A pokerbot.
    """

    def __init__(self) -> None:
        """
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        """
        self.num_shoves = 0
        self.num_rounds = 0
        self.log = []
        self.pre_computed_probs = pickle.load(open("python_skeleton/skeleton/pre_computed_probs.pkl", "rb")) 
        self.model = torch.load("python_skeleton/model.pt", weights_only=False, map_location=torch.device('cpu'))
        self.name = "Training Player"
        self.bankroll=0
        self.reward = 0
        pass

    def handle_new_round(self, game_state: GameState, round_state: RoundState, active: int) -> None:
        """
        Called when a new round starts. Called NUM_ROUNDS times.
        
        Args:
            game_state (GameState): The state of the game.
            round_state (RoundState): The state of the round.
            active (int): Your player's index.

        Returns:
            None
        """
        print("HI")
        #my_bankroll = game_state.bankroll # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        #game_clock = game_state.game_clock # the total number of seconds your bot has left to play this game
        #round_num = game_state.round_num # the round number from 1 to NUM_ROUNDS
        #my_cards = round_state.hands[active] # your cards
        #big_blind = bool(active) # True if you are the big blind
        self.log = []
        self.log.append("================================")
        self.log.append("new round")
        print("start")
        pass

    def handle_round_over(self, game_state: GameState, terminal_state: TerminalState, active: int, is_match_over: bool) -> Optional[str]:
        """
        Called when a round ends. Called NUM_ROUNDS times.

        Args:
            game_state (GameState): The state of the game.
            terminal_state (TerminalState): The state of the round when it ended.
            active (int): Your player's index.

        Returns:
            Your logs.
        """
        #my_delta = terminal_state.deltas[active] # your bankroll change from this round
        #previous_state = terminal_state.previous_state # RoundState before payoffs
        #street = previous_state.street # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[active] # your cards
        #opp_cards = previous_state.hands[1-active] # opponent's cards or [] if not revealed
        self.num_rounds += 1
        self.log.append("game over")
        self.log.append("================================\n")

        return self.log

    def get_action(self, observation) -> Action:
        """
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Args:
            observation (dict): The observation of the current state.
            {
                "legal_actions": List of the Actions that are legal to take.
                "street": 0, 1, or 2 representing pre-flop, flop, or river respectively
                "my_cards": List[str] of your cards, e.g. ["1s", "2h"]
                "board_cards": List[str] of the cards on the board
                "my_pip": int, the number of chips you have contributed to the pot this round of betting
                "opp_pip": int, the number of chips your opponent has contributed to the pot this round of betting
                "my_stack": int, the number of chips you have remaining
                "opp_stack": int, the number of chips your opponent has remaining
                "my_bankroll": int, the number of chips you have won or lost from the beginning of the game to the start of this round
                "min_raise": int, the smallest number of chips for a legal bet/raise
                "max_raise": int, the largest number of chips for a legal bet/raise
            }

        Returns:
            Action: The action you want to take.
        """

        inputs = GameData(observation["street"], [observation["my_pip"], observation["opp_pip"]], [observation["my_stack"], observation["opp_stack"]], [observation["my_cards"], observation["board_cards"]])
        state = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0) 
        trainRes = np.argmax(self.model.forward(state))
        
        if trainRes == 6:
            self.log.append(f"max raising to {observation["max_raise"]}")
            action = RaiseAction(observation["max_raise"])
        elif trainRes == 5:
            raise_amt = max(observation["max_raise"], 64)
            self.log.append(f"raising to {raise_amt}")
            action = RaiseAction(raise_amt)
        elif trainRes == 4:
            raise_amt = max(observation["min_raise"], 12)
            self.log.append(f"raising to {raise_amt}")
            action = RaiseAction(raise_amt)
        elif trainRes == 3:
            raise_amt = max(observation["min_raise"], 2)
            self.log.append(f"raising to {raise_amt}")
            action = RaiseAction(max(observation["min_raise"], 2))
        elif trainRes == 2:
            self.log.append("call actioning")
            action= CallAction()
        elif trainRes == 1:
            self.log.append("Check actioning")
            action = CheckAction()
        else:
            self.log.append("fold actioning")
            action = FoldAction()
        return action

if __name__ == '__main__':
    run_bot(TrainingPlayer(), parse_args())
