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

class Player(Bot):
    """
    A pokerbot.

    bluff probabilities are based off of hr9
    Changed bluff probabilities to make this more aggressive in its bluffing
    
    """
    def f(x):
        bluff = 0
        if x < 0.2:
            bluff =  0
        elif x < 0.4:
            bluff =  0.2
        elif x < 0.5:
            bluff = 0.4 
        elif x < 0.6:
            bluff = 0.6
        elif x < 0.8:
            bluff = 0.2
        else:
            bluff = 0
        return bluff


    def __init__(self,
                 bluff1 = f,
                 bluff2 = 1,
                 bluff3 = 0.1,
                 range1 = 300,
                 filter1 = 0.9,
                 range2 = 100,
                 filter2 = 0.7,
                 range3 = 10,
                 filter3 = 0.5,
                 filter4 = 0.2,
                 allin = 0.2,
                 allin2 = 0.1,
                 size = 0.75,
                 bluffsize = 0.35
                 ) -> None:
        """
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        """
        self.num_shoves = 0
        self.num_rounds = 0
        self.prop_raise = 0
        self.num_raises = 0
        self.num_bets = 1
        self.log = []
        self.bluff1 = bluff1
        self.bluff2 = bluff2
        self.bluff3 = bluff3
        self.size = size
        self.bluffsize = bluffsize
        self.range1 = range1
        self.range2 = range2
        self.range3 = range3
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3
        self.filter4 = filter4
        self.allin = allin
        self.allin2 = allin2
        self.name="player"
        self.bankroll=0
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
        self.num_bets += 1
        if continue_cost > 1:
            self.num_raises += 1
            if observation["opp_stack"] == 0:
                self.num_shoves += 1
        if (self.num_shoves / self.num_rounds >= self.allin and 
            (random.random() >= self.allin2) and
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
                if (opp_bet > self.range1):
                    equity = (equity - self.filter1) / (1 - self.filter1)
                elif (opp_bet > self.range2):
                    equity = (equity - self.filter2) / (1 - self.filter2)
                elif (opp_bet > self.range3):
                    equity = (equity - self.filter3) / (1 - self.filter3)
                else:
                    equity = (equity - self.filter4) / (1 - self.filter4)
                self.log.append(f"Adjusted equity: {equity}")
            if equity > 0.9 and RaiseAction in observation["legal_actions"]:
                action = RaiseAction(observation["max_raise"])
            elif equity > 0.8 and RaiseAction in observation["legal_actions"]:
                sizing = random.uniform(self.size*0.9, 
                                        self.size*1.1)
                raise_amount = min(int(pot_size*sizing), observation["max_raise"])
                raise_amount = max(raise_amount, observation["min_raise"])
                action = RaiseAction(raise_amount)
            elif CallAction in observation["legal_actions"] and equity >= pot_odds:
                if (random.random() < self.bluff1(equity) and RaiseAction in observation["legal_actions"]):
                    sizing = random.uniform(self.bluffsize*0.9, 
                                            self.bluffsize*1.1)
                    raise_amount = min(int(pot_size*sizing), observation["max_raise"])
                    raise_amount = max(raise_amount, observation["min_raise"])
                    action = RaiseAction(raise_amount)
                else:
                    action = CallAction()
            elif CheckAction in observation["legal_actions"]:
                if (random.random() > 1 - equity * self.bluff2 and 
                    equity > self.bluff3 and RaiseAction in observation["legal_actions"]):
                    sizing = random.uniform(self.bluffsize*0.9, 
                                            self.bluffsize*1.1)
                    raise_amount = min(int(pot_size*sizing), observation["max_raise"])
                    raise_amount = max(raise_amount, observation["min_raise"])
                    action = RaiseAction(raise_amount)
                else:
                    action = CheckAction()
            else:
                if (self.num_rounds >= 15 and self.num_raises / self.num_bets > 0.35 and 
                    random.random() > 1 - equity and 
                    equity > self.bluff3 and
                    CallAction in observation["legal_actions"]):
                    action = CallAction()
                else:
                    action = FoldAction()

        self.log.append(str(action) + "\n")

        return action

if __name__ == '__main__':
    run_bot(Player(), parse_args())
