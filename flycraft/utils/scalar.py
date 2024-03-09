import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def get_min_max_scalar(mins: np.ndarray, maxs: np.ndarray, feature_range: Tuple[int, int]=(0., 1.)):
    scalar = MinMaxScaler(feature_range=feature_range, clip=True, copy=True)
    return scalar.fit([mins, maxs])


