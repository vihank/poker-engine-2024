from argparse import ArgumentParser
from multiprocessing import Process
import subprocess
import os
import csv

from engine.engine import Game

excluded_files = set(["arnav_testing.py", 
                      "csv_scraper.py", 
                      "rl_bot.py",
                      "bluff_prob.py",
                      "prob_bot.py",
                      "all_in.py",
                      "antiallin_prob.py",
                      "prob_bot.py",
                      "trainingbot.py"])

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--docker", action="store_true", help="Running in containers")
    parser.add_argument("-p", type = str)

    return parser.parse_args()


def run_game_engine() -> None:
    """
    Runs the game engine process.
    """
    game = Game(False)
    game.run_match()


if __name__ == "__main__":
    args = parse_args()

    if args.docker:
        game_engine_process = Process(target=run_game_engine)
        game_engine_process.start()
        game_engine_process.join()
    else:
        for filename in os.listdir("./python_skeleton/"):
            if filename[-3:] == ".py" and filename != args.p and filename not in excluded_files:
                csvfile = './all_results.csv'
                with open(csvfile, 'a', newline="") as file:
                    csvwriter = csv.writer(file) # 2. create a csvwriter object
                    csvwriter.writerow([args.p, filename])
                file.close()
                player1_process = subprocess.Popen(
                    ["python", "python_skeleton/" + args.p, "--port", "50051"]
                )
                player2_process = subprocess.Popen(
                    ["python", "python_skeleton/" + filename, "--port", "50052"]
                )
                game_engine_process = Process(target=run_game_engine)
                game_engine_process.start()
                game_engine_process.join()
                player1_process.terminate()
                player2_process.terminate()
