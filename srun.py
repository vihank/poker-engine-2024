# simulate the game
# return the states of the game at every state (after every action)

# take some random players
import sys
sys.path.append('./python_skeleton')

from collections import defaultdict, deque
import os
import csv
from typing import Deque, List


from python_skeleton.bluff_prob import BluffPlayer
from python_skeleton.all_in import AllInPlayer
from python_skeleton.past_hist import PastPlayer
from python_skeleton.aggressive_prob_bot import ProbPlayer
from python_skeleton.player1 import Player
from python_skeleton.trainingbot import TrainingPlayer, DQN
from python_skeleton.hr1 import RangePlayer1
from python_skeleton.hr2 import RangePlayer2
from python_skeleton.hr3 import RangePlayer3
from python_skeleton.hr4 import RangePlayer4
from python_skeleton.hr5 import RangePlayer5
from python_skeleton.hr6 import RangePlayer6
from python_skeleton.hr7 import RangePlayer7
from python_skeleton.hr8 import RangePlayer8
from python_skeleton.hr9 import RangePlayer9
from python_skeleton.hr10 import RangePlayer10



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

from csv_scraper import GameData, HandData

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
        self.logging = True
        self.printing = True
        self.ret = False
        self.isfirst = True

    def log_round_state(self, round_state: RoundState):
        """
        Logs the current state of the round.
        """

        if round_state.street == 0 and round_state.button == 0:
            self.log.append(f"{self.players[0].name} posts the blind of {SMALL_BLIND}")
            self.log.append(f"{self.players[1].name} posts the blind of {BIG_BLIND}")
            self.log.append(f"{self.players[0].name} dealt {round_state.hands[0]}")
            self.log.append(f"{self.players[1].name} dealt {round_state.hands[1]}")

            self._create_csv_row(round_state, self.players[0].name, "posts blind", SMALL_BLIND)
            self._create_csv_row(round_state, self.players[1].name, "posts blind", BIG_BLIND)

        elif round_state.street > 0 and round_state.button == 1:
            # log the pot every street
            pot = STARTING_STACK - round_state.stacks[0] + STARTING_STACK - round_state.stacks[1]
            self.log.append(f"{STREET_NAMES[round_state.street]} Board: {round_state.board} Pot: {pot}")

    def log_action(
        self, player_name: str, action: Action, round_state: RoundState
    ) -> None:
        """
        Logs an action taken by a player.
        """
        if isinstance(action, FoldAction):
            self.log.append(f"{player_name} folds")
            self._create_csv_row(round_state, player_name, "fold", None)
        elif isinstance(action, CallAction):
            self.log.append(f"{player_name} calls")
            self._create_csv_row(round_state, player_name, "call", None)
        elif isinstance(action, CheckAction):
            self.log.append(f"{player_name} checks")
            self._create_csv_row(round_state, player_name, "check", None)
        else:  # isinstance(action, RaiseAction)
            self.log.append(f"{player_name} bets {str(action.amount)}")
            self._create_csv_row(round_state, player_name, "bets", action.amount)

    def log_terminal_state(self, round_state: TerminalState) -> None:
        """
        Logs the terminal state of a round, including outcomes.
        """
        previous_state = round_state.previous_state
        if FoldAction not in previous_state.legal_actions():  # idk why this is needed
            self.log.append(f"{self.players[0].name} shows {previous_state.hands[0]}")
            self.log.append(f"{self.players[1].name} shows {previous_state.hands[1]}")
        self.log.append(f"{self.players[0].name} awarded {round_state.deltas[0]}")
        self.log.append(f"{self.players[1].name} awarded {round_state.deltas[1]}")
        self.log.append(f"{self.players[0].name} Bankroll: {self.players[0].bankroll}")
        self.log.append(f"{self.players[1].name} Bankroll: {self.players[1].bankroll}")

    def run_round(self, last_round: bool, gameData):
        """
        Runs one round of poker (1 hand).
        """
        print("RUNNING ROUND")
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        deck = ShortDeck()
        deck.shuffle()
        hands = [deck.deal(2), deck.deal(2)]
        self.players[0].handle_new_round(None, None, None)
        
        self.players[1].handle_new_round(None, None, None)

        episode = HandData(isfirst = self.isfirst, hand = hands[0] if self.isfirst else hands[1], opp_hand=hands[1] if self.isfirst else hands[0])
        actions=[]
    
        round_state = RoundState(0, 0, pips, stacks, hands, [], deck, None)
        self.new_actions = [deque(), deque()]
        turn = defaultdict(list)
        turn["street"] = 0
        while not isinstance(round_state, TerminalState):
            if self.logging: self.log_round_state(round_state)
            if turn["street"] < round_state.street:
                actions.append(turn)
                turn=defaultdict(list)
                turn["street"] = round_state.street

            active = round_state.button % 2
            player = self.players[active]

            action = fixactions(player.get_action(request_obs_translate(round_state, active)))
            
            turn["public_cards"] = round_state.board
            mvmt, amt= actionTranslate(action)
            if player.name == self.original_players[0].name:    
                turn["your_action"].append(mvmt)
                turn["ActionAmt"].append(amt)
                if self.ret: yield turn
            else:
                turn["opp_action"].append(mvmt)
                turn["opp_amount"].append(amt)

            action = self._validate_action(action, round_state, player.name)
            if self.logging: self.log_action(player.name, action, round_state)

            self.new_actions[1 - active].append(action)
            round_state = round_state.proceed(action)

        actions.append(turn)

        episode.add_turns(actions)

        board = round_state.previous_state.board
        for index, (player, delta) in enumerate(zip(self.players, round_state.deltas)):

            game_state = GameState(bankroll=player.bankroll, round_num=self.round_num, game_clock=0)
            pRoundState = TerminalState(tuple(round_state.deltas), board)

            player.handle_round_over(game_state,pRoundState, active, last_round)
            player.bankroll += delta

            if player.name == self.original_players[0].name:
                episode.add_reward(player.bankroll)

        if self.logging: 
            self.log_terminal_state(round_state)
            num_raise = [getattr(player, "num_raises", 0)/ getattr(player, "num_rounds", 1) for player in self.players]
            self.log.append(f"{self.original_players[0].name} numraise something: {num_raise[0]}")
            self.log.append(f"{self.original_players[1].name} numraise something: {num_raise[1]}")
            episode.add_num_raise(num_raise)



    def run_match(self, bots):
        """
        Runs one match of poker.
        """
        gameData = GameData() if self.ret else None
        if self.printing: print("Starting the Poker Game...")
        self.players = [
            bots[0], bots[1]
        ]
        player_names = [self.players[0].name, self.players[1].name]

        if self.printing: print(f"Player 1: {player_names[0]}, Player 2: {player_names[1]}")

        if self.printing: print("Starting match...")
        self.original_players = self.players.copy()
        for self.round_num in range(1, NUM_ROUNDS + 1):
            if self.round_num % 100 == 0:
                if self.printing: print(f"Starting round {self.round_num}...")
            self.log.append(f"\nRound #{self.round_num}")

            ret = self.run_round((self.round_num == NUM_ROUNDS), gameData)

            self.players = self.players[::-1]  # Alternate the dealer
            self.isfirst != self.isfirst

        self.log.append(f"{self.original_players[0].name} Bankroll: {self.original_players[0].bankroll}")
        self.log.append(f"{self.original_players[1].name} Bankroll: {self.original_players[1].bankroll}")

        if self.logging: self._finalize_log()


    def _finalize_log(self) -> None:
        """
        Finalizes the game log, writing it to a file and uploading it.
        """
        csv_filename = f"{GAME_LOG_FILENAME}thin.csv"
        self._upload_or_write_file(self.csvlog, csv_filename, is_csv=True)

        log_filename = f"{GAME_LOG_FILENAME}thig.txt"
        self._upload_or_write_file(self.log, log_filename)


    def _upload_or_write_file(self, content, base_filename, is_csv=False):
        filename = self._get_unique_filename(base_filename)
        
        filename = os.path.join(LOGS_DIRECTORY, filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if self.printing: print(f"Writing {filename}")
        mode = "w"
        newline = "" if is_csv else None
        with open(filename, mode, newline=newline) as file:
            if is_csv:
                writer = csv.writer(file)
                writer.writerows(content)
            else:
                file.write("\n".join(content))

    @staticmethod
    def _get_unique_filename(base_filename):
        file_idx = 1
        filename, ext = os.path.splitext(base_filename)
        unique_filename = base_filename
        while os.path.exists(unique_filename):
            unique_filename = f"{filename}_{file_idx}{ext}"
            file_idx += 1
        return unique_filename

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

    def _create_csv_row(
        self, round_state: RoundState, player_name: str, action: str, action_amt: int
    ) -> None:
        self.csvlog.append([
            self.round_num,
            round_state.street,
            player_name,
            action,
            action_amt if action_amt else "",
            " ".join(round_state.hands[0] if self.round_num % 2 == 1 else round_state.hands[1]),
            " ".join(round_state.hands[1] if self.round_num % 2 == 1 else round_state.hands[0]),
            " ".join(round_state.board),
            self.original_players[0].bankroll,
        ])


if __name__ == '__main__':
    game = Game(logging=True, ret=True)
    for strat in [Player()]:
        print(f"{list(game.run_match([TrainingPlayer(), strat]))[-1]} against {strat.name}")
    # probs a good idea to check if the names are as you exepect coming out of this thing
    # print("Game Over")