"""
Simple example pokerbot, written in Python.
"""

import itertools
import random
import pickle
from typing import Optional

from skeleton.actions import Action, CallAction, CheckAction, FoldAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.evaluate import evaluate

class RangePlayer2(Bot):
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
        #my_bankroll = game_state.bankroll # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        #game_clock = game_state.game_clock # the total number of seconds your bot has left to play this game
        #round_num = game_state.round_num # the round number from 1 to NUM_ROUNDS
        #my_cards = round_state.hands[active] # your cards
        #big_blind = bool(active) # True if you are the big blind
        self.num_rounds += 1
        self.log = []
        self.log.append("================================")
        self.log.append("new round")
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
        self.log.append("game over")
        self.log.append("================================\n")

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
            (random.random() >= 0.1) and
            self.num_rounds >= 5):
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
                opp_bet = observation["opp_pip"]
                if (opp_bet > 350):
                    equity = (equity - 0.9) / (1 - 0.9)
                elif (opp_bet > 15):
                    equity = (equity - 0.7) / (1 - 0.7)
                else:
                    equity = (equity - 0.5) / (1 - 0.5)
                self.log.append(f"Adjusted equity: {equity}")
            if equity > 0.9 and RaiseAction in observation["legal_actions"]:
                action = RaiseAction(observation["max_raise"])
            elif equity > 0.8 and RaiseAction in observation["legal_actions"]:
                raise_amount = min(int(pot_size*0.75), observation["max_raise"])
                raise_amount = max(raise_amount, observation["min_raise"])
                action = RaiseAction(raise_amount)
            elif CallAction in observation["legal_actions"] and equity >= pot_odds:
                if (random.random() > 0.9):
                    raise_amount = min(int(pot_size*0.75), observation["max_raise"])
                    raise_amount = max(raise_amount, observation["min_raise"])
                    action = RaiseAction(raise_amount)
                else:
                    action = CallAction()
            elif CheckAction in observation["legal_actions"]:
                if (random.random() > 1 - equity * 1/2 and equity > 0.2):
                    raise_amount = min(int(pot_size*0.75), observation["max_raise"])
                    raise_amount = max(raise_amount, observation["min_raise"])
                    action = RaiseAction(raise_amount)
                else:
                    action = CheckAction()
            else:
                action = FoldAction()

        self.log.append(str(action) + "\n")

        return action

if __name__ == '__main__':
    run_bot(RangePlayer2(), parse_args())
