import pickle
import pandas as pd

pre_computed_probs = pickle.load(open("python_skeleton/skeleton/pre_computed_probs.pkl", "rb"))

print(pre_computed_probs)