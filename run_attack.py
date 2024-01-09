import argparse
import shutil
import warnings

import torch
from foolbox import PyTorchModel
from foolbox.attacks import LinfPGD, FGSM, LinfBasicIterativeAttack
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from PGD_freq import *
from model import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
window_size = 2048


def optimize(data, model, label, adv_distortion, th_batch, psd_max_batch, lr_stage, num_iter_stage, init_alpha):
    delta = adv_distortion.clone().detach().requires_grad_(True)
    th_loss = torch.tensor([[np.inf] * 10]).reshape((10, 1)).to(device)
    alpha = torch.ones((10, 1)) * init_alpha
    alpha = alpha.to(device)
    final_alpha = torch.zeros(10)
    final_adv = torch.zeros_like(data)
    optimizer = torch.optim.Adam([delta], lr=lr_stage)
    min_th = -np.inf
    for i in range(num_iter_stage):
        new_input = delta + data
        new_input_stego = torch.clamp(new_input, 0, float('inf'))
        stego_output = model(new_input_stego)
        _, predicted = torch.max(stego_output, 1)
        cross_loss = criterion(stego_output, label).to(device)
        th_loss_temp = compute_loss_th(delta, window_size, th_batch, psd_max_batch).to(device)
        total_loss = -cross_loss + (alpha * th_loss_temp).sum()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        th_loss_output = th_loss_temp.cpu().detach().numpy()
        alpha_output = alpha.cpu().detach().numpy()

        for ii in range(10):
            if predicted[ii] != label[ii]:
                if th_loss_temp[ii] < th_loss[ii]:
                    th_loss[ii] = th_loss_temp[ii]
                    final_alpha[ii] = alpha[ii]
                    final_adv[ii] = new_input[ii]
            if i % 20 == 0:
                alpha[ii] *= 1.2
            if i % 20 == 0 and predicted[ii] == label[ii]:
                alpha[ii] *= 0.8
                alpha[ii] = max(alpha[ii], min_th)
            print('Iteration [{}/{}], th_loss: {}, '
                  'alpha: {}'.format(ii + 1, i + 1, th_loss_output[ii], alpha_output[ii]))
            if i == num_iter_stage - 1 and (final_adv[ii] == 0).all():
                final_adv[ii] = new_input[ii]
    return final_adv, th_loss, final_alpha


def main(args):
    warnings.filterwarnings("ignore")

    if args.dataset == 'Urban8K':
        data_path = 'stft_data'
        psd_path = 'th_bath_and_psd_max_batch'
        num_classes = 10
        if args.model == 'VGG13':
            model = VGG13(num_classes=num_classes)
            model.load_state_dict(torch.load('model_state/VGG13.pth'))
        else:
            model = VGG16(num_classes=num_classes)
            model.load_state_dict(torch.load('model_state/VGG16.pth'))
    else:
        data_path = 'esc_data'
        psd_path = 'esc50_psd'
        num_classes = 50
        if args.model == 'VGG13':
            model = VGG13(num_classes=num_classes)
            model.load_state_dict(torch.load('model_state/VGG13_esc50.pth'))
        else:
            model = VGG16(num_classes=num_classes)
            model.load_state_dict(torch.load('model_state/VGG16_esc50.pth'))

    model.to(device)
    model.eval()

    f_model = PyTorchModel(model, bounds=(0, float('inf')))

    if args.attack_method == 'PGD':
        attack = LinfPGD(steps=40, rel_stepsize=1.0 / 40, random_start=True)

    elif args.attack_method == 'FGSM':
        attack = FGSM(random_start=True)

    elif args.attack_method == 'BIM':
        attack = LinfBasicIterativeAttack(steps=40, rel_stepsize=1.0 / 40)

    elif args.attack_method == 'PGD_freq':
        attack = PGD_freq(steps=40, rel_stepsize=1.0 / 40, random_start=True)

    save_path = os.path.join(args.save_path, args.dataset, args.model, args.attack_method)

    attack_rates = []
    for class_id in tqdm(range(10, num_classes), leave=True):
        print(f'The current class under attack is the {class_id}th class')

        data = torch.load(os.path.join(data_path, f'class{class_id}', 'real', 'correct_real.pt'))
        label = torch.load(os.path.join(data_path, f'class{class_id}', 'label', 'correct_labels.pt'))

        if args.attack_method == 'PGD_freq_psy':
            th_batch = torch.load(os.path.join(psd_path, f'class{class_id}', 'th_bath.pt'))
            th_batch = th_batch.permute(0, 2, 1).to(device)
            psd_max_batch = torch.load(os.path.join(psd_path, f'class{class_id}', 'psd_max_batch.pt'))

            adv_example = torch.load(
                os.path.join(str(args.save_path), str(args.dataset), str(args.model), 'PGD_freq',
                             f'class_{class_id}_adv_audio.pt'))

            adv = adv_example - data
            adv_, th_loss, final_alpha = optimize(data, model, label, adv, th_batch, psd_max_batch, args.lr_stage,
                                                  args.num_iter_stage, args.alpha)
            torch.save(adv_, os.path.join(str(save_path), f'class_{class_id}_adv_audio.pt'))

            y_true = []
            y_pred = []
            outputs = model(adv_)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(label.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
            attack_success = 1 - accuracy_score(y_true, y_pred)
            attack_rates.append(attack_success)
        elif args.attack_method == 'PGD_psy':
            th_batch = torch.load(os.path.join(psd_path, f'class{class_id}', 'th_batch.pt'))
            # th_batch = torch.stack(th_batch)
            th_batch = th_batch.permute(0, 2, 1).to(device)
            psd_max_batch = torch.load(os.path.join(psd_path, f'class{class_id}', 'psd_max_batch.pt'))
            # psd_max_batch = torch.stack(psd_max_batch)

            adv_example = torch.load(
                os.path.join(str(args.save_path), str(args.dataset), str(args.model), 'PGD',
                             f'class_{class_id}_adv_audio.pt'))

            adv = adv_example - data.clone()

            adv_, th_loss, final_alpha = optimize(data, model, label, adv, th_batch, psd_max_batch,
                                                  args.lr_stage, args.num_iter_stage, args.alpha)

            torch.save(adv_, os.path.join(str(save_path), f'class_{class_id}_adv_audio.pt'))

            y_true = []
            y_pred = []
            outputs = model(adv_)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(label.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
            attack_success = 1 - accuracy_score(y_true, y_pred)
            attack_rates.append(attack_success)
        else:
            if args.attack_method == 'PGD_freq':
                freqs = librosa.fft_frequencies(sr=22050, n_fft=2048)
                weights = librosa.A_weighting(freqs)
                normalized_weights = (weights - min(weights)) / (max(weights) - min(weights))
                scaling_factor = 0.2
                eps = args.epsilon + (1 - normalized_weights) * scaling_factor - np.mean(
                    (1 - normalized_weights) * scaling_factor)
                epsilon = eps.max()
            else:
                epsilon = args.epsilon
            _, adv_example, success = attack(f_model, data, label, epsilons=epsilon)
            attack_rates.append(success.float().mean().item())
            torch.save(adv_example, os.path.join(str(save_path), f'class_{class_id}_adv_audio.pt'))

    # 计算指标
    snr = calculate_average_metric_for_classes(data_path, save_path, metric='snr')
    stoi = calculate_average_metric_for_classes(data_path, save_path, metric='stoi')
    succ = sum(attack_rates) / len(attack_rates)

    with open('success_metric.txt', 'a') as file:
        print(args.dataset, args.model, args.attack_method, succ, snr, stoi, file=file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--dataset', type=str, default='ESC-50', help='Dataset name')
    parser.add_argument('--model', type=str, default='VGG13', help='Model name')
    parser.add_argument('--lr_stage', type=float, default=0.001, help='Learning rate for optimize')
    parser.add_argument('--num_iter_stage', type=int, default=500, help='Number of iterations for optimize')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value')
    parser.add_argument('--attack_method', type=str, default='PGD_psy', help='Attack method')
    parser.add_argument('--save_path', type=str, default='adv_example')
    parser.add_argument('--alpha', type=float, default=0.07)
    args = parser.parse_args()
    main(args)
