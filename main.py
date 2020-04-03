import heartpy
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from heartpy.datautils import rolling_mean
from sklearn import preprocessing

from utils.filters import bandpass_butter

# from model.apply_model import apply_model_df


def crop_center(img, ww=400, hh=500):
    h, w, _ = img.shape
    w_st = w // 2 - (ww // 2)
    h_st = h // 2 - (hh // 2)
    return img[h_st:h_st + ww, w_st:w_st + hh]


def img_to_signal(img, crop=False):
    if crop:
        img = crop_center(img)

    # fn = np.sum
    fn = np.mean

    r, g, b = img[:, :, 0], img[:, :, 0], img[:, :, 0]
    out = [fn(r), fn(g), fn(b)]
    return out


def process_data(x, fps=30):
    print(x.shape)

    x = preprocessing.scale(x)
    x = bandpass_butter(x, cut_low=1, cut_high=2, rate=fps, order=2)

    rol_mean = rolling_mean(x, windowsize=5, sample_rate=fps)
    wd = heartpy.peakdetection.detect_peaks(x, rol_mean, ma_perc=20, sample_rate=fps)
    peaks = wd['peaklist']
    rri = np.diff(peaks)
    rri = rri * 1 / fps * 1000
    hr = 6e4 / rri

    # df = pd.DataFrame()
    # df['rri'] = [rri]
    # res = apply_model_df(df)
    # print(f"res: {res}")

    # m, wd = heartpy.analysis.calc_breathing(wd['RR_list'], x, sample_rate=fps)
    # print(f"breath rate: {round(m['breathingrate'], 3)}")

    fig, axs = plt.subplots(3, 1, figsize=(17, 9))
    axs = axs.ravel()
    axs[0].plot(x, '-b', label='x')
    axs[0].plot(peaks, x[peaks], 'r.', label='peaks')
    axs[1].plot(rri, '-r.', label='rri')
    axs[2].plot(hr, '-r.', label='hr')
    for ax in axs:
        ax.legend()
    plt.show()


def main():
    fp = 'data/MOV_0079.mp4'
    vid = imageio.get_reader(fp, 'ffmpeg')
    metadata = vid.get_meta_data()
    print(f'metadata: {metadata}')
    fps = metadata['fps']
    rgb = [img_to_signal(img) for img in vid.iter_data()]
    rgb = np.vstack(rgb)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    process_data(r, fps)


if __name__ == "__main__":
    main()
