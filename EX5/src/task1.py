import pandas as pd
import numpy as np


if __name__ == '__main__':
    # TASK 1.1
    # read the file
    linear_func_data = pd.read_csv("../data/linear_function_data.txt", sep=" ", header=None, dtype=np.float64)

