import argparse

import librosa
import numpy as np
import torch
from scipy import signal
import os


def compute_PSD_matrix(stft, window_size):
    """
    Compute the Power Spectral Density (PSD) matrix.

    Parameters:
    - stft: Short Time Fourier Transform (STFT) matrix.
    - window_size: Size of the window used for STFT.

    Returns:
    - PSD: Transformed PSD matrix.
    - psd_max: Maximum PSD value.
    """
    # Scaling the STFT result
    stft_result = np.sqrt(8.0 / 3.) * stft

    # Normalize by window size
    stft_result = stft_result / window_size

    # Further normalization
    stft_tensor = stft_result / window_size

    # Calculate the maximum PSD value
    psd_max = torch.max(stft_tensor * stft_tensor)

    # Compute the PSD in decibels
    psd = 10 * torch.log10(stft_tensor * stft_tensor + 1e-20)

    # Transform PSD to a new scale
    PSD = 96 - torch.max(psd) + psd

    return PSD, psd_max


def Bark(f):
    """
    Compute the Bark scale values for a given frequency.

    Parameters:
    - f: Input frequency.

    Returns:
    - Bark scale value corresponding to the input frequency.
    """
    # Compute the Bark scale using the specified formula
    return 13 * torch.atan(0.00076 * f) + 3.5 * torch.atan((f / 7500.0) ** 2)


def quiet(f):
    """
    Compute the quiet threshold for a given frequency.

    Parameters:
    - f: Input frequency.

    Returns:
    - Quiet threshold corresponding to the input frequency.
    """
    # Calculate the quiet threshold using the specified formula
    thresh = 3.64 * torch.pow(f * 0.001, -0.8) - 6.5 * torch.exp(
        -0.6 * torch.pow(0.001 * f - 3.3, 2)) + 0.001 * torch.pow(0.001 * f, 4) - 12

    return thresh


def two_slops(bark_psd, delta_TM, bark_maskee):
    """
    Compute the two slopes for each tone mask.

    Parameters:
    - bark_psd: Bark scale PSD values for tone masks.
    - delta_TM: Delta threshold of hearing for each tone mask.
    - bark_maskee: Bark scale values for maskee.

    Returns:
    - Ts: List of computed thresholds for each tone mask.
    """
    Ts = []

    # Iterate over each tone mask
    for tone_mask in range(bark_psd.size(0)):
        bark_masker = bark_psd[tone_mask, 0]
        dz = bark_maskee - bark_masker

        # Find the index where dz becomes non-negative
        zero_index = torch.argmax((dz > 0).float())

        # Compute the slopes based on the conditions
        sf = torch.zeros_like(dz)
        sf[:zero_index] = 27 * dz[:zero_index]
        sf[zero_index:] = (-27 + 0.37 * torch.maximum(bark_psd[tone_mask, 1] - 40, torch.tensor(0.0))) * dz[zero_index:]

        # Calculate the final threshold
        T = bark_psd[tone_mask, 1] + delta_TM[tone_mask] + sf
        Ts.append(T)

    return Ts


def compute_th(PSD, barks, ATH, freqs):
    """
    Compute the threshold of hearing for a given PSD spectrum.

    Parameters:
    - PSD: Power Spectral Density spectrum.
    - barks: Bark scale values corresponding to the frequency bins.
    - ATH: Absolute Threshold of Hearing.
    - freqs: Frequencies corresponding to the PSD spectrum.

    Returns:
    - theta_x: Computed threshold of hearing.
    """
    length = PSD.size(0)

    # Find local maxima indices in the PSD spectrum
    masker_index = torch.tensor(signal.argrelextrema(PSD.cpu().numpy(), np.greater)[0])

    # Remove endpoints from the indices
    masker_index = masker_index[masker_index != 0]
    masker_index = masker_index[masker_index != length - 1]
    num_local_max = masker_index.size(0)

    # Compute power levels for local maxima and adjacent points
    p_k = torch.pow(10, PSD[masker_index] / 10.)
    p_k_prev = torch.pow(10, PSD[masker_index - 1] / 10.)
    p_k_post = torch.pow(10, PSD[masker_index + 1] / 10.)
    P_TM = 10 * torch.log10(p_k_prev + p_k + p_k_post)

    # Create a matrix to store bark scale, power levels, and index of local maxima
    bark_psd = torch.zeros(num_local_max, 3)
    bark_psd[:, 0] = barks[masker_index]
    bark_psd[:, 1] = P_TM
    bark_psd[:, 2] = masker_index.float()

    # Process overlapping maskers and remove less significant ones
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

    # Compute delta threshold based on Bark scale
    delta_TM = 1 * (-6.025 - 0.275 * bark_psd[:, 0])

    # Compute the final thresholds using two slopes approach
    Ts = two_slops(bark_psd, delta_TM, barks)

    # Stack the computed thresholds
    if not Ts:
        Ts = torch.zeros(len(barks))
    else:
        Ts = torch.stack(Ts)

    # Calculate the sum of power levels and add Absolute Threshold of Hearing
    theta_x = torch.sum(torch.pow(10, Ts / 10.), dim=0) + torch.pow(10, ATH / 10.)

    return theta_x


def generate_th(stft, fs, window_size):
    """
    Generate the threshold of hearing for each time frame in an audio signal.

    Parameters:
    - stft: Short Time Fourier Transform (STFT) of the audio signal.
    - fs: Sampling rate of the audio signal.
    - window_size: Size of the window used for STFT.

    Returns:
    - theta_xs: Computed thresholds of hearing for each time frame.
    - psd_max: Maximum PSD value.
    """
    # Compute the Power Spectral Density (PSD) matrix and maximum PSD value
    PSD, psd_max = compute_PSD_matrix(stft=stft, window_size=2048)

    # Compute frequencies and corresponding Bark scale values
    freqs = librosa.core.fft_frequencies(sr=fs, n_fft=window_size)
    barks = Bark(torch.tensor(freqs))

    # Initialize the Absolute Threshold of Hearing (ATH) with negative infinity
    ATH = torch.full((len(barks),), float('-inf'))

    # Find the index where Bark scale exceeds 1
    bark_ind = torch.argmax((barks > 1).float())

    # Set ATH values for indices beyond the threshold
    ATH[bark_ind:] = quiet(torch.tensor(freqs[bark_ind:]))

    # Compute thresholds of hearing for each time frame using compute_th function
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

    # Define paths and parameters based on the dataset
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

    # Loop over each class in the dataset
    for class_id in range(n_class):
        th_batch = []
        psd_max_batch = []

        # Load real audio data for the current class
        data = torch.load(os.path.join(data_path, f'class{class_id}', 'real', 'correct_real.pt'))

        # Generate PSD and thresholds for a batch of audio samples
        for i in range(batch_size):
            th, psd_max = generate_th(data[i].squeeze(), args.sample_rate, args.window_size)
            th_batch.append(th)
            psd_max_batch.append(psd_max)

        # Stack the computed thresholds and PSD max values
        th_batch = torch.stack(th_batch)
        psd_max_batch = torch.stack(psd_max_batch)

        # Create a directory to save the results for the current class
        path = os.path.join(save_path, f'class{class_id}')
        os.makedirs(path, exist_ok=True)

        # Save the computed thresholds and PSD max values
        torch.save(th_batch, os.path.join(path, 'th_batch.pt'))
        torch.save(psd_max_batch, os.path.join(path, 'psd_max_batch.pt'))

