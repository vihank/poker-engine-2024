# simulate the game
# return the states of the game at every state (after every action)

# take some random players
import sys
sys.path.append('./python_skeleton')

from collections import defaultdict, deque
import os
import csv
import random
from typing import Deque, List

from engine.evaluate import evaluate

from engine.roundstate import RoundState
from engine.evaluate import ShortDeck
from engine.actions import (
    STREET_NAMES,
    Action,
    CallAction,
    CheckAction,
    FoldAction,
    RaiseAction,
    TerminalState,
)
from engine.config import (
    BIG_BLIND,
    BOT_LOG_FILENAME,
    GAME_LOG_FILENAME,
    LOGS_DIRECTORY,
    NUM_ROUNDS,
    PLAYER_1_DNS,
    PLAYER_1_NAME,
    PLAYER_2_DNS,
    PLAYER_2_NAME,
    SMALL_BLIND,
    STARTING_STACK,
    upload_logs,
    add_match_entry,
)
from python_skeleton.skeleton.bot import Bot
from python_skeleton.skeleton.states import GameState
from skeleton.actions import (
    CallAction as sCallAction, 
    CheckAction as sCheckAction,
    FoldAction as sFoldAction,
    RaiseAction as sRaiseAction,
)

class GameData():
    def __init__(self, street, stacks, pips, hand):
        self.state = (street, 400 - stacks[0], 400 - stacks[1], pips[0], pips[1], *self.extract_hand(hand), evaluate(hand[0], hand[1]) / 1000 )

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




class HandData():
    __allowed = ("isfirst", "reward", "hand", "opp_hand", "turns", "bankroll")
    def __init__(self, **kwarg):
        for k, v in kwarg.items():
            assert(k in self.__class__.__allowed)
            setattr(self, k, v)

    def add_turns(self, turns):
        self.turns = turns


def actionsfix(action): # input type is engine
    if action == FoldAction:
        return sFoldAction
    elif action == CallAction:
        return sCallAction
    elif action == CheckAction:
        return sCheckAction
    else:
        return sRaiseAction
    
def fixactions(action): #input type is skeleton
    if isinstance(action, sFoldAction):
        return FoldAction()
    elif isinstance(action, sCallAction):
        return CallAction()
    elif isinstance(action, sCheckAction):
        return CheckAction()
    else:
        return RaiseAction(action.amount)
    
def actionTranslate(action):
    if isinstance(action, FoldAction):
        return "fold", 0
    elif isinstance(action, CallAction):
        return "call", 0
    elif isinstance(action, CheckAction):
        return "check", 0
    else:
        return "bets", 1


def request_obs_translate(round_state, active):
    obs = {}
    obs["legal_actions"] = set([actionsfix(a) for a in round_state.legal_actions()])
    obs["street"] = round_state.street
    obs["my_cards"] = round_state.hands[0]
    obs["board_cards"] = list(round_state.board)
    obs["my_pip"] = round_state.pips[active]
    obs["opp_pip"] = round_state.pips[1 - active]
    obs["my_stack"] = round_state.stacks[active]
    obs["opp_stack"] = round_state.stacks[1 - active]
    obs["my_bankroll"] = 0
    obs["min_raise"] = round_state.raise_bounds()[0]
    obs["max_raise"] = round_state.raise_bounds()[1]

    return obs

class Game:
    def __init__(self, logging=False, printing=False, ret=False) -> None:
        self.players: List[Bot] = []
        self.log: List[str] = [
            f"CMU Poker Bot Game - {PLAYER_1_NAME} vs {PLAYER_2_NAME}"
        ]
        self.csvlog: List[str] = [
            [
                "Round",
                "Street",
                "Team",
                "Action",
                "ActionAmt",
                "Team1Cards",
                "Team2Cards",
                "AllCards",
                "Bankroll",
            ]
        ]
        self.new_actions: List[Deque[Action]] = [deque(), deque()]
        self.round_num = 0
        self.logging = logging
        self.printing = printing
        self.ret = ret
        self.isfirst = True

    
    def run_round(self, last_round: bool):
        """
        Runs one round of poker (1 hand).
        """
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        deck = ShortDeck()
        deck.shuffle()
        hands = [deck.deal(2), deck.deal(2)]
        self.players[0].handle_new_round(None, None, None)
        self.players[1].handle_new_round(None, None, None)
        self.isfirst = (True if random.random() > 0.5 else False)
        episode = HandData(isfirst = self.isfirst, hand = hands[0] if self.isfirst else hands[1], opp_hand=hands[1] if self.isfirst else hands[0])
        actions=[]

        round_state = RoundState(0, 0, pips, stacks, hands, [], deck, None)
        self.new_actions = [deque(), deque()]
        turn = defaultdict(list)
        turn["street"] = 0
        
        active = round_state.button % 2
        player = self.players[active]
        
        if active == 1:

            print(player.name)
            action = fixactions(player.get_action(request_obs_translate(round_state, active)))

            turn["public_cards"] = round_state.board
            mvmt, amt = actionTranslate(action)
            turn["your_action"].append(mvmt)
            turn["ActionAmt"].append(amt)
            
            action = self._validate_action(action, round_state, player.name)

        return round_state

    def run_turn(self, round_state, player_action):
        turn = defaultdict(list)
        turn["street"] = round_state.street

        # Player 1
        active = round_state.button % 2
        player = self.players[0]
        action = fixactions(player.get_action(request_obs_translate(round_state, active), player_action))
        
        turn["public_cards"] = round_state.board
        mvmt, amt= actionTranslate(action)
        turn["your_action"].append(mvmt)
        turn["ActionAmt"].append(amt)
        action = self._validate_action(action, round_state, player.name)
        
        self.new_actions[1 - active].append(action)
        round_state = round_state.proceed(action)
        if isinstance(round_state, TerminalState):
          return int(round_state[0][0])

        if turn["street"] < round_state.street:
            turn = defaultdict(list)
            turn["street"] = round_state.street

        # Player 2
        player = self.players[1]
        action = fixactions(player.get_action(request_obs_translate(round_state, active)))
        
        turn["public_cards"] = round_state.board
        mvmt, amt= actionTranslate(action)

        turn["opp_action"].append(mvmt)
        turn["opp_amount"].append(amt)

        action = self._validate_action(action, round_state, player.name)
        self.new_actions[1 - active].append(action)
        round_state = round_state.proceed(action)

        return round_state

    def run_match(self, bots):
        """
        Runs one match of poker.
        """
        if self.printing: print("Starting the Poker Game...")
        self.players = [
            bots[0], bots[1]
        ]
        player_names = [self.players[0].name, self.players[1].name]


    def _validate_action(
        self, action: Action, round_state: RoundState, player_name: str
    ) -> Action:
        """
        Validates an action taken by a player, ensuring it's legal given the current round state.
        If the action is illegal, defaults to a legal action (Check if possible, otherwise Fold).

        Args:
            action (Action): The action attempted by the player.
            round_state (RoundState): The current state of the round.
            player_name (str): The name of the player who took the action.

        Returns:
            Action: The validated (or corrected) action.
        """
        legal_actions = (
            round_state.legal_actions()
            if isinstance(round_state, RoundState)
            else {CheckAction}
        )
        if isinstance(action, RaiseAction):
            amount = int(action.amount)
            min_raise, max_raise = round_state.raise_bounds()
            active = round_state.button % 2
            continue_cost = round_state.pips[1 - active] - round_state.pips[active]
            if RaiseAction in legal_actions and min_raise <= amount <= max_raise:
                return action
            elif CallAction in legal_actions and amount > continue_cost:
                self.log.append(f"{player_name} attempted illegal RaiseAction with amount {amount}")
                return CallAction()
            else:
                self.log.append(f"{player_name} attempted illegal RaiseAction with amount {amount}")
        elif type(action) in legal_actions:
            return action
        else:
            self.log.append(f"{player_name} attempted illegal {type(action).__name__}")

        return CheckAction() if CheckAction in legal_actions else FoldAction()