import numpy as np
from scipy import signal
from scipy.ndimage.filters import maximum_filter, minimum_filter, uniform_filter, median_filter

"""
1 dimension signal filters
"""


def lowpass_butter(arr, cut, rate, order=2):
    nyq = 0.5 * rate
    cut = cut / nyq
    b, a = signal.butter(order, cut, btype='low', analog=False, output='ba')
    out = signal.filtfilt(b, a, arr)
    return out


def highpass_butter(arr, cut, rate, order=2):
    nyq = 0.5 * rate
    cut = cut / nyq
    b, a = signal.butter(order, cut, btype='high', analog=False, output='ba')
    out = signal.filtfilt(b, a, arr)
    return out


def bandpass_butter(arr, cut_low, cut_high, rate, order=2):
    nyq = 0.5 * rate
    low = cut_low / nyq
    high = cut_high / nyq
    b, a = signal.butter(order, [low, high], btype='band', analog=False, output='ba')
    out = signal.filtfilt(b, a, arr)
    return out


def lowpass_fir(arr, cut, fs, order):
    nyq = 0.5 * fs
    cut = cut / nyq
    h = signal.firwin(order, cut, pass_zero='lowpass')
    out = signal.lfilter(h, 1.0, arr)
    # out = signal.filtfilt(h, 1.0, arr)
    return out


def highpass_fir(arr, cut, fs, order):
    nyq = 0.5 * fs
    cut = cut / nyq
    h = signal.firwin(order, cut, pass_zero='highpass')
    out = signal.lfilter(h, 1.0, arr)
    # out = signal.filtfilt(h, 1.0, arr)
    return out


def bandpass_fir(arr, cut_low, cut_high, fs, order):
    nyq = 0.5 * fs
    low, high = cut_low / nyq, cut_high / nyq
    h = signal.firwin(order, [low, high], pass_zero='bandpass')
    out = signal.lfilter(h, 1.0, arr)
    # out = signal.filtfilt(h, 1.0, arr)
    return out


def bandpass_cheby1(arr, cut_low, cut_high, fs, order, rp):
    nyq = 0.5 * fs
    low, high = cut_low / nyq, cut_high / nyq
    b, a = signal.cheby1(order, rp, [low, high], btype='bandpass', analog=False, output='ba')
    out = signal.filtfilt(b, a, arr)
    return out


def lowpass_cheby1(arr, cut, fs, order, rp):
    nyq = 0.5 * fs
    cut = cut / nyq
    b, a = signal.cheby1(order, rp, cut, btype='lowpass', analog=False, output='ba')
    out = signal.filtfilt(b, a, arr)
    return out


def highpass_cheby1(arr, cut, fs, order, rp):
    nyq = 0.5 * fs
    cut = cut / nyq
    b, a = signal.cheby1(order, rp, cut, btype='highpass', analog=False, output='ba')
    out = signal.filtfilt(b, a, arr)
    return out


def med_filter(arr, kernel_size=5):
    out = signal.medfilt(arr, kernel_size)
    return out


def med_filter_2(arr, kernel_size=5):
    out = median_filter(arr, kernel_size)
    return out


def max_filter(arr, kernel_size=5):
    out = maximum_filter(arr, kernel_size)
    return out


def min_filter(arr, kernel_size=5):
    out = minimum_filter(arr, kernel_size)
    return out


def mean_filter(arr, kernel_size=5):
    out = uniform_filter(arr, kernel_size)
    return out


def moving_average(arr, window=5):
    # 不同的 weights 对应不同的滤波方式
    weights = np.repeat(1.0, window) / window
    out = np.convolve(arr, weights, 'valied')
    return out


def wiener_filter(arr, kernel_size=5):
    out = signal.wiener(arr, kernel_size)
    return out


def savgol_filter(arr, window_length=5, polyorder=2, *argv):
    out = signal.savgol_filter(arr, window_length, polyorder, *argv)
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, num=1000)
    y = np.sin(20 * x)
    y += np.random.randn(len(x)) * 0.5
    # y += np.random.uniform(low=0.01, high=0.3, size=(len(y), ))

    # out = med_filter(y)
    out = savgol_filter(y)

    plt.plot(y, '-b', label='y')
    plt.plot(out, '-r', label='filtered')
    plt.legend()
    plt.show()
