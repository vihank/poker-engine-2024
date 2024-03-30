from srun import Game

from python_skeleton.player import Player
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
from python_skeleton.aggressive_prob_bot import ProbPlayer
import numpy as np


if __name__ == '__main__':
    game = Game(logging=True, ret=True)

    paramRange = np.arange(0.4,0.9, 0.05)
    bestParam = None
    past_res=-float('inf')
    for filter2 in [0.5]:# paramRange:
        print(f"new filter dropped {filter2}")
        res = []
        for strat in [ProbPlayer(), RangePlayer1(), RangePlayer2(), RangePlayer3(), RangePlayer4(), RangePlayer5(), RangePlayer6(), RangePlayer7(), RangePlayer8(), RangePlayer9(), RangePlayer10()]:
            res += [x.reward for x in game.run_match([Player(filter2=filter2), strat])]
            

        print(res)
        curr_res = np.mean(res)

        if curr_res > past_res:
            bestParam = filter2
            past_res = curr_res

    print(f"the best param is {bestParam} with a average reward of {past_res}")