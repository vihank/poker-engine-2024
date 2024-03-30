"""
Simple example pokerbot, written in Python.
"""

import itertools
import random
import pickle
from typing import Optional
import numpy as np

from antiallin_prob import AntiAllInPlayer
from bluff_prob import BluffPlayer
from handranging_bluff0 import RangePlayer1
from handranging_bluff1 import RangePlayer2
from handranging_bluff2 import RangePlayer3


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
        
        self.weighting = np.array([0.03, 0.33, 0.3, 0.5])
        self.c = 0.5

        self.bots = {0: AntiAllInPlayer(), 1: RangePlayer1(), 2: RangePlayer2(), 3: RangePlayer3()}

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
        self.num_turns += 1

        soft_rewards = np.exp(self.payoffs) / np.sum(np.exp(self.payoffs))
        ucb = 1 + self.c * np.sqrt(np.log(self.num_turns) * np.reciprocal(self.times_chosen + 1e-6))
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
        self.log.append("=================v6==============\n")
        self.payoffs[self.cur_bot] = (self.payoffs[self.cur_bot] * self.times_chosen[self.cur_bot] + terminal_state.deltas[active]) / (self.times_chosen[self.cur_bot] + 1)
        self.times_chosen[self.cur_bot] += 1
        self.bots[self.cur_bot].handle_round_over(game_state, terminal_state, active, is_match_over)
        self.log.append(f"bot used: {self.cur_bot}, freq: {self.times_chosen[self.cur_bot]}")
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
        return self.bots[self.cur_bot].get_action(observation)

if __name__ == '__main__':
    run_bot(Player(), parse_args())
