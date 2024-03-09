import pandas as pd
import numpy as np
from numpy.fft import fft
from typing import List
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils.smoothness import fourier

def smoothness_measure_by_delta(traj: pd.DataFrame, measure_columns: List[str]):
    for c in measure_columns:
        assert c in traj.columns, f"{c}不是轨迹中的字段!!!"

    res = []
    for c in measure_columns:
        tmp = sum([np.abs(a-b) for a, b in zip(traj[c][:-1], traj[c][1:])]) / len(traj)
        res.append(tmp)
    
    return res

def smoothness_measure_by_fft(traj: pd.DataFrame, measure_columns: List[str]):
    for c in measure_columns:
        assert c in traj.columns, f"{c}不是轨迹中的字段!!!"
    
    def calc_sm_by_fft(actions):
        freqs, amps = fourier.fourier_transform(actions, 0.1)
        sm = fourier.smoothness(amps)
        return sm   

    res = []
    for c in measure_columns:
        res.append(calc_sm_by_fft(list(traj[c])))
    
    return res


if __name__ == "__main__":
    a = np.arange(12).reshape((4,3))
    df = pd.DataFrame(data=a, columns=['a', 'b', 'c'])
    res = smoothness_measure_by_delta(df, ['a', 'b'])
    print(res)
