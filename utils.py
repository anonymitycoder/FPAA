import os

import librosa
import numpy as np
import pystoi
import torch


def transform(x, window_size, psd_max_ori):
    scale = 8. / 3.
    z = scale * (x / window_size)
    psd = torch.square(z)
    PSD = torch.pow(torch.tensor(10.), 9.6) / psd_max_ori * psd
    return PSD


def compute_loss_th(delta, window_size, th_batch, psd_max_batch):
    f = torch.nn.ReLU()
    losses = []
    for i in range(delta.size(0)):
        logits_delta = transform(delta[i, :], window_size, psd_max_batch[i])
        loss_th = f(logits_delta.squeeze() - th_batch[i]).mean()
        losses.append(loss_th)
    loss_th = torch.stack(losses)
    return loss_th


def calculate_snr(clean_wav, noisy_wav, epsilon=1e-10):
    noise_power = np.sum((clean_wav - noisy_wav) ** 2)
    clean_power = np.sum(clean_wav ** 2)
    snr = 10 * np.log10(clean_power / (noise_power + epsilon))
    return snr


def restore_audio_from_magnitude_and_phase(magnitude, phase):
    stft_matrix = magnitude * np.exp(1j * phase)
    audio = librosa.istft(stft_matrix)
    return audio


def calculate_stoi(reference_waveform, degraded_waveform, sample_rate=22050):
    stoi_score = pystoi.stoi(reference_waveform, degraded_waveform, sample_rate, extended=False)
    return stoi_score


def calculate_average_metric_for_classes(clean_data_path, noisy_data_path, metric='snr'):
    list_metric = []
    for class_id in range(10):
        phase_angle = torch.load(os.path.join(clean_data_path, f'class{class_id}', 'imaginary', 'correct_imaginary.pt'))
        clean_wav = torch.load(os.path.join(clean_data_path, f'class{class_id}', 'real', 'correct_real.pt'))
        noisy_wav = torch.load(os.path.join(noisy_data_path, f'class_{class_id}_adv_audio.pt'))
        value = []
        for clean, attacked, phase_angle in zip(clean_wav, noisy_wav, phase_angle):
            clean = clean.squeeze().cpu().numpy()
            attacked = attacked.squeeze().detach().cpu().numpy()
            phase_angle = phase_angle.squeeze().cpu().numpy()

            clean_audio = restore_audio_from_magnitude_and_phase(clean, phase_angle)
            attacked_audio = restore_audio_from_magnitude_and_phase(attacked, phase_angle)

            if metric == 'snr':
                value.append(calculate_snr(clean_audio, attacked_audio))
            else:
                value.append(calculate_stoi(clean_audio, attacked_audio))
        list_metric.append(sum(value) / len(value))
    return list_metric
