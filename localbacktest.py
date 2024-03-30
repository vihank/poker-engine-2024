from srun import Game
from python_skeleton.all_in import AllInPlayer
from python_skeleton.bluff_prob import BluffPlayer

if __name__ == "__main__":
    game = Game(logging=False, printing=False, ret=False)
    for _ in range(3):
        game.run_match([AllInPlayer(), BluffPlayer()])
