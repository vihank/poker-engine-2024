"""
Simple example pokerbot, written in Python.
"""

import itertools
import random
import pickle
import math
import numpy as np
from typing import Optional

from arnav_prob import ArnavPlayer
from prob_bot import ProbPlayer
from all_in import AllInPlayer

from skeleton.actions import Action, CallAction, CheckAction, FoldAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.evaluate import evaluate

class Player(Bot):
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
        
        self.weighting = np.array([0.01, 0.5, 1.0])
        self.c = 0.5
        
        self.bots = {0: AllInPlayer(), 1: ProbPlayer(), 2: ArnavPlayer()}
        
        self.cur_bot = 0
        self.num_bots = len(self.bots)
        
        self.times_chosen = np.zeros(self.num_bots)
        self.payoffs = np.zeros(self.num_bots)
        self.num_turns = 0
        print("initialized")

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
        #my_bankroll = game_state.bankroll # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        #game_clock = game_state.game_clock # the total number of seconds your bot has left to play this game
        #round_num = game_state.round_num # the round number from 1 to NUM_ROUNDS
        #my_cards = round_state.hands[active] # your cards
        #big_blind = bool(active) # True if you are the big blind
        self.log = []
        self.log.append("================================")
        self.log.append("new round")
        self.num_turns = 0

        soft_rewards = np.exp(self.payoffs) / np.sum(np.exp(self.payoffs))
        ucb = 1 + self.c * np.sqrt(np.log(self.num_turns) * np.reciprocal(self.times_chosen))
        probs = soft_rewards * self.weighting * ucb
        norm_probs = probs / np.sum(probs)
        self.cur_bot = np.random.choice(list(range(self.num_bots)), p = norm_probs)

        self.bots[self.cur_bot].handle_new_round(game_state, round_state, active)

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
        self.log.append("game over")
        self.log.append("================================\n")
        self.num_turns += 1
        self.payoffs[self.cur_bot] = (self.payoffs[self.cur_bot] * self.times_chosen[self.cur_bot] + terminal_state.deltas[active]) / (self.times_chosen[self.cur_bot] + 1)
        self.times_chosen[self.cur_bot] += 1
        self.bots[self.cur_bot].handle_round_over(game_state, terminal_state, active, is_match_over)

        return self.log

    def get_action(self, observation: dict) -> Action:
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
<<<<<<< HEAD
        
        return self.bots[self.cur_bot].get_action(observation)
=======
        my_contribution = STARTING_STACK - observation["my_stack"] # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - observation["opp_stack"] # the number of chips your opponent has contributed to the pot
        pot_size = my_contribution + opp_contribution # the number of chips in the pot
        continue_cost = observation["opp_pip"] - observation["my_pip"] # the number of chips needed to stay in the pot

        self.log.append("My cards: " + str(observation["my_cards"]))
        self.log.append("Board cards: " + str(observation["board_cards"]))
        self.log.append("My stack: " + str(observation["my_stack"]))
        self.log.append("My contribution: " + str(my_contribution))
        self.log.append("My bankroll: " + str(observation["my_bankroll"]))

        # Original probability calculation
        # leftover_cards = [f"{rank}{suit}" for rank in "123456789" for suit in "shd" if f"{rank}{suit}" not in observation["my_cards"] + observation["board_cards"]]
        # possible_card_comb = list(itertools.permutations(leftover_cards, 4 - len(observation["board_cards"])))
        # possible_card_comb = [observation["board_cards"] + list(c) for c in possible_card_comb]
        # result = map(lambda x: evaluate(observation["my_cards"], x[:2]) > evaluate(x[:2], x[2:]), possible_card_comb)
        # prob = sum(result) / len(possible_card_comb)

        # Use pre-computed probability calculation
        equity = self.pre_computed_probs['_'.join(sorted(observation["my_cards"])) + '_' + '_'.join(sorted(observation["board_cards"]))]
        pot_odds = continue_cost / (pot_size + continue_cost)

        self.log.append(f"Equity: {equity}")
        self.log.append(f"Pot odds: {pot_odds}")

        # If the villain raised, adjust the probability
        if continue_cost > 1:
            if observation["opp_stack"] == 0:
                self.num_shoves += 1
        if (self.num_shoves / self.num_rounds >= 0.2 and 
            (random.random() >= 0.1)):
            if equity > 0.51 and (RaiseAction in observation["legal_actions"]):
                action = RaiseAction(observation["max_raise"])
            elif equity > 0.51 and (CallAction in observation["legal_actions"]):
                action = CallAction()
            elif CheckAction in observation["legal_actions"]:
                action = CheckAction()
            else:
                action = FoldAction()
        else:
            if continue_cost > 1:
                equity = (equity - 0.5) / 0.5
                self.log.append(f"Adjusted equity: {equity}")
            if equity > 0.9 and RaiseAction in observation["legal_actions"]:
                action = RaiseAction(observation["max_raise"])
            elif equity > 0.8 and RaiseAction in observation["legal_actions"]:
                raise_amount = min(int(pot_size*0.75), observation["max_raise"])
                raise_amount = max(raise_amount, observation["min_raise"])
                action = RaiseAction(raise_amount)
            elif CallAction in observation["legal_actions"] and equity >= pot_odds:
                action = CallAction()
            elif CheckAction in observation["legal_actions"]:
                action = CheckAction()
            else:
                action = FoldAction()

        self.log.append(str(action) + "\n")

        return action
>>>>>>> dd8a972a1188ceb55b7e9306aac012f3fb657805

if __name__ == '__main__':
    run_bot(Player(), parse_args())
