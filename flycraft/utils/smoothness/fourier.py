"""comes from CAPS

paper: Regularizing Action Policies for Smooth Control with Reinforcement Learning

code: http://ai.bu.edu/caps

Returns:
    _type_: _description_
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
# from scipy.fft import fft
from scipy.fftpack import fft
from scipy import stats
import os.path as osp

def smoothness(amplitudes):
    return np.mean(amplitudes*normalized_freqs(amplitudes))

def center_of_mass(amplitudes, freqs):
    return np.sum(freqs * amplitudes) / sum(amplitudes)

def save_smoothness(amplitudes, fpath):
    smoothness_score = smoothness(amplitudes)
    print(smoothness_score)
    joblib.dump(smoothness_score, osp.join(fpath, 'smoothness.pkl'))


def fourier_transform(actions, T=0.002):
    N = len(actions)
    # print(N)
    x = np.linspace(0.0, N*T, N)
    y = actions
    yf = fft(y)
    freq = np.linspace(0.0, 1.0/(2.0*T), N//2)
    amplitudes = 2.0/N * np.abs(yf[0:N//2])
    # plt.plot(freq, amplitudes)
    return freq, amplitudes

def cut_data(actionss, ep_lens):
    median = int(np.median(ep_lens))
    print("median:", median)
    same_len = map(lambda x: x[:median], filter(lambda x: len(x) >= median, actionss))
    return same_len

def to_array_truncate(l):
    min_len = min(map(len, l))
    return np.array(list(map(lambda x: x[min_len:], l)))


def combine(fouriers):
    freqs = fouriers[0][0]
    amplitudess = np.array(list(map(lambda x: x[1], fouriers)))

    amplitudes = np.mean(amplitudess, axis=0)
    return freqs, amplitudes

def from_actions(actionss, ep_lens):
    fouriers = list(map(fourier_transform, cut_data(actionss, ep_lens)))
    return combine(fouriers)

def normalized_freqs(amplitudes):
    return np.linspace(0, 1, amplitudes.shape[0])

def plot(freqs, amplitudes, amplitudes_std=None, title=None):
    f_m, ax_m = plt.subplots(1,1,figsize=(7,5), sharey=True, sharex=True)
    # amp_std = np.std(amplitudes)
    # amp_mean = np.mean(amplitudes)
    # print("t:", amp_std, amp_mean)
    print(freqs[-1])
    plt.fill_between(freqs, 0, amplitudes, where=amplitudes >= 0, facecolor='#003c69')
    if not (amplitudes_std is None):
        y = amplitudes + amplitudes_std
        plt.fill_between(freqs, 0, y, where=y >= 0, facecolor='#003c69', alpha=0.6)
    ax_m.set_ylabel('Amplitude')
    if title:
        ax_m.set_title(title)
    ax_m.set_xlabel("Hz")
    return f_m, ax_m

