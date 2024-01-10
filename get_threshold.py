import argparse

import librosa
import numpy as np
import torch
from scipy import signal
import os


def compute_PSD_matrix(stft, window_size):
    stft_result = np.sqrt(8.0 / 3.) * stft
    stft_result = stft_result / window_size
    stft_tensor = stft_result / window_size
    psd_max = torch.max(stft_tensor * stft_tensor)
    psd = 10 * torch.log10(stft_tensor * stft_tensor + 1e-20)
    PSD = 96 - torch.max(psd) + psd
    return PSD, psd_max


def Bark(f):
    return 13 * torch.atan(0.00076 * f) + 3.5 * torch.atan((f / 7500.0) ** 2)


def quiet(f):
    thresh = 3.64 * torch.pow(f * 0.001, -0.8) - 6.5 * torch.exp(
        -0.6 * torch.pow(0.001 * f - 3.3, 2)) + 0.001 * torch.pow(0.001 * f, 4) - 12
    return thresh


def two_slops(bark_psd, delta_TM, bark_maskee):
    Ts = []
    for tone_mask in range(bark_psd.size(0)):
        bark_masker = bark_psd[tone_mask, 0]
        dz = bark_maskee - bark_masker
        zero_index = torch.argmax((dz > 0).float())
        sf = torch.zeros_like(dz)
        sf[:zero_index] = 27 * dz[:zero_index]
        sf[zero_index:] = (-27 + 0.37 * torch.maximum(bark_psd[tone_mask, 1] - 40, torch.tensor(0.0))) * dz[zero_index:]
        T = bark_psd[tone_mask, 1] + delta_TM[tone_mask] + sf
        Ts.append(T)
    return Ts


def compute_th(PSD, barks, ATH, freqs):
    length = PSD.size(0)
    masker_index = torch.tensor(signal.argrelextrema(PSD.cpu().numpy(), np.greater)[0])

    masker_index = masker_index[masker_index != 0]
    masker_index = masker_index[masker_index != length - 1]
    num_local_max = masker_index.size(0)

    p_k = torch.pow(10, PSD[masker_index] / 10.)
    p_k_prev = torch.pow(10, PSD[masker_index - 1] / 10.)
    p_k_post = torch.pow(10, PSD[masker_index + 1] / 10.)
    P_TM = 10 * torch.log10(p_k_prev + p_k + p_k_post)

    bark_psd = torch.zeros(num_local_max, 3)
    bark_psd[:, 0] = barks[masker_index]
    bark_psd[:, 1] = P_TM
    bark_psd[:, 2] = masker_index.float()

    i = 0
    while i < bark_psd.size(0):
        next_i = i + 1
        while next_i < bark_psd.size(0) and bark_psd[next_i, 0] - bark_psd[i, 0] < 0.5:
            if quiet(freqs[int(bark_psd[i, 2])]) > bark_psd[i, 1]:
                bark_psd = torch.cat((bark_psd[:i], bark_psd[i + 1:]), dim=0)
                continue

            if bark_psd[i, 1] < bark_psd[next_i, 1]:
                bark_psd = torch.cat((bark_psd[:i], bark_psd[i + 1:]), dim=0)
            else:
                bark_psd = torch.cat((bark_psd[:next_i], bark_psd[next_i + 1:]), dim=0)
            next_i += 1

        i += 1

    delta_TM = 1 * (-6.025 - 0.275 * bark_psd[:, 0])
    Ts = two_slops(bark_psd, delta_TM, barks)

    if not Ts:
        Ts = torch.zeros(len(barks))
    else:
        Ts = torch.stack(Ts)

    theta_x = torch.sum(torch.pow(10, Ts / 10.), dim=0) + torch.pow(10, ATH / 10.)

    return theta_x


def generate_th(stft, fs, window_size):
    PSD, psd_max = compute_PSD_matrix(stft=stft, window_size=2048)

    freqs = librosa.core.fft_frequencies(sr=fs, n_fft=window_size)
    barks = Bark(torch.tensor(freqs))

    ATH = torch.full((len(barks),), float('-inf'))
    bark_ind = torch.argmax((barks > 1).float())
    ATH[bark_ind:] = quiet(torch.tensor(freqs[bark_ind:]))

    theta_xs = []
    for i in range(PSD.shape[1]):
        theta_xs.append(compute_th(PSD[:, i], barks, ATH, torch.tensor(freqs)))
    theta_xs = torch.stack(theta_xs)

    return theta_xs, psd_max


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='urban8K', help='Dataset name')
    parser.add_argument('--psd_save_path', type=str, default='urban_psd', help='Path to save PSD')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate')
    parser.add_argument('--window_size', type=int, default=2048, help='Window size')
    args = parser.parse_args()

    psd_path = 'psd'

    if args.dataset == 'urban8K':
        batch_size = 50
        n_class = 10
        data_path = 'urban8k_data'
        save_path = psd_path + 'urban_psd'
    else:
        batch_size = 10
        n_class = 50
        data_path = 'esc_data'
        save_path = psd_path + 'esc_psd'

    for class_id in range(n_class):
        th_batch = []
        psd_max_batch = []
        data = torch.load(os.path.join(data_path, f'class{class_id}', 'real', 'correct_real.pt'))
        for i in range(batch_size):
            th, psd_max = generate_th(data[i].squeeze(), args.sample_rate, args.window_size)
            th_batch.append(th)
            psd_max_batch.append(psd_max)
        th_batch = torch.stack(th_batch)
        psd_max_batch = torch.stack(psd_max_batch)

        path = os.path.join(save_path, f'class{class_id}')
        os.makedirs(path, exist_ok=True)

        torch.save(th_batch, os.path.join(path, 'th_batch.pt'))
        torch.save(psd_max_batch, os.path.join(path, 'psd_max_batch.pt'))
