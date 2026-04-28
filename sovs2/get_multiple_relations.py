import pandas as pd
import numpy as np
from itertools import combinations
import os


def get_csv_list(path):
    csv_files = []
    dir = os.listdir(path)
    for file in dir:
        if file.endswith(".csv"):
            csv_files.append(file)
    
    return csv_files


def loop_files(path):
    csv_list = get_csv_list(path)
    for name in csv_list:
        df = pd.read_csv(name)
        compare(df)


def compare(df):
    combs = list(combinations(list(df.columns), 2))
    for comb in combs:
        x = df[comb[0]].to_numpy()
        y = df[comb[1]].to_numpy()
        



if __name__ == "__main__":
    data_folder = "./data"

    print(get_csv_list(data_folder))
    