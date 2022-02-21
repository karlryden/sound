import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

from linear import nu, step


M = read_csv('./data/microphones.csv', header=None).to_numpy()

if __name__ == '__main__':
    print(M)