from srun import Game
from python_skeleton.all_in import AllInPlayer
from python_skeleton.bluff_prob import BluffPlayer
from python_skeleton.trainingbot import TrainingPlayer

if __name__ == "__main__":
    game = Game(logging=False, printing=False, ret=True)
    for episodeinfo in game.run_math([TrainingPlayer(), BluffPlayer()]):
        trainingfunction(episodeinfo)
