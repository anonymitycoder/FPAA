import argparse
import pystoi
import torch
import os
import numpy as np
import librosa


# Function to calculate Signal-to-Noise Ratio (SNR)
def calculate_snr(clean_wav, noisy_wav, epsilon=1e-10):
    noise_power = np.sum((clean_wav - noisy_wav) ** 2)
    clean_power = np.sum(clean_wav ** 2)
    snr = 10 * np.log10(clean_power / (noise_power + epsilon))
    return snr


# Function to restore audio from magnitude and phase information
def restore_audio_from_magnitude_and_phase(magnitude, phase):
    stft_matrix = magnitude * np.exp(1j * phase)
    audio = librosa.istft(stft_matrix)
    return audio


# Function to calculate Short Time Objective Intelligibility (STOI)
def calculate_stoi(reference_waveform, degraded_waveform, sample_rate=22050):
    stoi_score = pystoi.stoi(reference_waveform, degraded_waveform, sample_rate, extended=False)
    return stoi_score


# Function to calculate average metric for classes
def calculate_average_metric_for_classes(clean_data_path, noisy_data_path, metric='snr'):
    list_metric = []
    for class_id in range(10):
        phase_angle = torch.load(
            os.path.join(clean_data_path, f'class{class_id}', 'imaginary', 'correct_imaginary.pt'))
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
    return sum(list_metric) / len(list_metric)



if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Urban8K', help='Dataset name')
    parser.add_argument('--model', type=str, default='VGG13', help='Model name')
    parser.add_argument('--attack_method', type=str, default='PGD_psy', help='Attack method')
    parser.add_argument('--adv_example_path', type=str, default='adv_example')
    args = parser.parse_args()

    # Set the data path based on the dataset
    if args.dataset == 'Urban8K':
        data_path = 'urban8K_data'
    else:
        data_path = 'esc_data'

    # Construct the path to the adversarial examples
    adv_path = os.path.join(args.adv_example_path, args.dataset, args.model, args.attack_method)

    # Calculate average SNR and STOI metrics for the classes
    snr = calculate_average_metric_for_classes(data_path, adv_path, metric='snr')
    stoi = calculate_average_metric_for_classes(data_path, adv_path, metric='stoi')

    # Append the results to a file
    with open('success_metric.txt', 'a') as file:
        print(args.dataset, args.model, args.attack_method, snr, stoi, file=file)
