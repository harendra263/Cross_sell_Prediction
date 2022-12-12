from imblearn.under_sampling import RandomUnderSampling
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from . import dispatcher


TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = os.environ.get("FOLD")
MODEL = int(os.environ.get("MODEL"))

FOLD_MAPPING = {
    0: [1, 2, 3, 4, 5],
    1: [0, 2, 3, 4, 5],
    2: [0, 1, 3, 4, 5],
    3: [0, 1, 2, 4, 5],
    4: [0, 1, 2, 3, 5],
    5: [0, 1, 2, 3, 4]
}


if __name__ == "__main__":
    df = pd.read_csv("input/train_folds.csv")



# train test split


# Imbalance Dataset
