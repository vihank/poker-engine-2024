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
from generic_handranger import RangePlayerK


from skeleton.actions import Action, CallAction, CheckAction, FoldAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.evaluate import evaluate

