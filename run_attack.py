import argparse
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
    # Clone the adversarial distortion tensor and make it require gradients for optimization
    delta = adv_distortion.clone().detach().requires_grad_(True)

    # Initialize tensor for threshold loss with infinity values
    th_loss = torch.tensor([[np.inf] * 25]).reshape((25, 1)).to(device)

    # Initialize alpha tensor with specified initial value
    alpha = torch.ones((25, 1)) * init_alpha
    alpha = alpha.to(device)

    # Initialize tensors to store final results
    final_alpha = torch.zeros(25)
    final_adv = torch.zeros_like(data)

    # Initialize Adam optimizer for adversarial perturbation
    optimizer = torch.optim.Adam([delta], lr=lr_stage)

    # Set a minimum threshold value
    min_th = -np.inf

    # Main optimization loop
    for i in range(num_iter_stage):
        # Create a new input by adding adversarial perturbation to the original data
        new_input = delta + data

        # Ensure the new input is within valid pixel intensity range
        new_input_stego = torch.clamp(new_input, 0, float('inf'))

        # Get model predictions on stego images
        stego_output = model(new_input_stego)
        _, predicted = torch.max(stego_output, 1)

        # Compute the cross-entropy loss
        cross_loss = criterion(stego_output, label).to(device)

        # Compute threshold loss using the provided function
        th_loss_temp = compute_loss_th(delta, window_size, th_batch, psd_max_batch).to(device)

        # Calculate the total loss as a combination of cross-entropy and threshold losses
        total_loss = -cross_loss + (alpha * th_loss_temp).sum()

        # Perform optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Convert tensors to numpy arrays for printing
        th_loss_output = th_loss_temp.cpu().detach().numpy()
        alpha_output = alpha.cpu().detach().numpy()

        # Update threshold loss, alpha, and adversarial example based on conditions
        for ii in range(25):
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
            # Print iteration details
            print(
                'Iteration [{}/{}], th_loss: {}, alpha: {}'.format(ii + 1, i + 1, th_loss_output[ii], alpha_output[ii]))

            # Set final adversarial example if conditions are met
            if i == num_iter_stage - 1 and (final_adv[ii] == 0).all():
                final_adv[ii] = new_input[ii]

    # Return final results
    return final_adv, th_loss, final_alpha


def main(args):
    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Set data paths and number of classes based on the dataset
    if args.dataset == 'Urban8K':
        data_path = 'urban8K_data'
        psd_path = 'psd/urban_psd'
        num_classes = 10
        # Load pre-trained model based on the chosen architecture
        if args.model == 'VGG13':
            model = VGG13(num_classes=num_classes)
            model.load_state_dict(torch.load('model_state/VGG13.pth'))
        else:
            model = VGG16(num_classes=num_classes)
            model.load_state_dict(torch.load('model_state/VGG16.pth'))
    else:
        data_path = 'esc_data'
        psd_path = 'psd/esc_psd'
        num_classes = 50
        # Load pre-trained model based on the chosen architecture
        if args.model == 'VGG13':
            model = VGG13(num_classes=num_classes)
            model.load_state_dict(torch.load('model_state/VGG13_esc50.pth'))
        else:
            model = VGG16(num_classes=num_classes)
            model.load_state_dict(torch.load('model_state/VGG16_esc50.pth'))

    # Move the model to the specified device and set it to evaluation mode
    model.to(device)
    model.eval()

    # Create a PyTorchModel wrapper for the model
    f_model = PyTorchModel(model, bounds=(0, float('inf')))

    # Choose the attack method based on the provided argument
    if args.attack_method == 'PGD':
        attack = LinfPGD(steps=40, rel_stepsize=1.0 / 40, random_start=True)
    elif args.attack_method == 'FGSM':
        attack = FGSM(random_start=True)
    elif args.attack_method == 'BIM':
        attack = LinfBasicIterativeAttack(steps=40, rel_stepsize=1.0 / 40)
    elif args.attack_method == 'PGD_freq':
        attack = PGD_freq(steps=40, rel_stepsize=1.0 / 40, random_start=True)

    # Set the save path for the adversarial examples
    save_path = os.path.join(args.save_path, args.dataset, args.model, args.attack_method)

    # Initialize a list to store attack success rates for each class
    attack_rates = []

    # Iterate over a range of classes (from 3 to 9)
    for class_id in tqdm(range(3, 10), leave=True):
        print(f'The current class under attack is the {class_id}th class')

        # Load data and labels for the specified class
        data = torch.load(os.path.join(data_path, f'class{class_id}', 'real', 'correct_real.pt'))[25:]
        label = torch.load(os.path.join(data_path, f'class{class_id}', 'label', 'correct_labels.pt'))[25:]

        # Handle specific attack methods with additional preprocessing
        if args.attack_method == 'PGD_freq_psy':
            # Load additional data for PSD-based attacks
            th_batch = torch.load(os.path.join(psd_path, f'class{class_id}', 'th_bath.pt'))
            th_batch = th_batch.permute(0, 2, 1).to(device)
            psd_max_batch = torch.load(os.path.join(psd_path, f'class{class_id}', 'psd_max_batch.pt'))

            # Load pre-generated adversarial examples
            adv_example = torch.load(
                os.path.join(str(args.save_path), str(args.dataset), str(args.model), 'PGD_freq',
                             f'class_{class_id}_adv_audio.pt'))

            # Compute adversarial example and optimize it
            adv = adv_example - data
            adv_, th_loss, final_alpha = optimize(data, model, label, adv, th_batch, psd_max_batch, args.lr_stage,
                                                  args.num_iter_stage, args.alpha)

            # Save the optimized adversarial example
            torch.save(adv_, os.path.join(str(save_path), f'class_{class_id}_adv_audio.pt'))

            # Evaluate the attack success rate
            y_true = []
            y_pred = []
            outputs = model(adv_)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(label.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
            attack_success = 1 - accuracy_score(y_true, y_pred)
            attack_rates.append(attack_success)
        # Handle another specific attack method with additional preprocessing
        elif args.attack_method == 'PGD_psy':
            th_batch = torch.load(os.path.join(psd_path, f'class{class_id}', 'th_batch.pt'))[25:]
            th_batch = torch.stack(th_batch)
            th_batch = th_batch.permute(0, 2, 1).to(device)
            psd_max_batch = torch.load(os.path.join(psd_path, f'class{class_id}', 'psd_max_batch.pt'))[25:]
            psd_max_batch = torch.stack(psd_max_batch)

            adv_example = torch.load(
                os.path.join(str(args.save_path), str(args.dataset), str(args.model), 'PGD',
                             f'class_{class_id}_adv_audio.pt'))[25:]

            adv = adv_example - data.clone()

            adv_, th_loss, final_alpha = optimize(data, model, label, adv, th_batch, psd_max_batch,
                                                  args.lr_stage, args.num_iter_stage, args.alpha)

            torch.save(adv_, os.path.join(str(save_path), f'class_{class_id}_adv_audio2.pt'))

            y_true = []
            y_pred = []
            outputs = model(adv_)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(label.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
            attack_success = 1 - accuracy_score(y_true, y_pred)
            attack_rates.append(attack_success)
        else:
            # Handle other attack methods
            if args.attack_method == 'PGD_freq':
                # Preprocess for frequency-based attacks
                freqs = librosa.fft_frequencies(sr=22050, n_fft=2048)
                weights = librosa.A_weighting(freqs)
                normalized_weights = (weights - min(weights)) / (max(weights) - min(weights))
                scaling_factor = 0.2
                eps = args.epsilon + (1 - normalized_weights) * scaling_factor - np.mean(
                    (1 - normalized_weights) * scaling_factor)
                epsilon = eps.max()
            else:
                epsilon = args.epsilon
            # Generate adversarial examples and compute success rate
            _, adv_example, success = attack(f_model, data, label, epsilons=epsilon)
            attack_rates.append(success.float().mean().item())
            torch.save(adv_example, os.path.join(str(save_path), f'class_{class_id}_adv_audio.pt'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--dataset', type=str, default='Urban8K', help='Dataset name')
    parser.add_argument('--model', type=str, default='VGG13', help='Model name')
    parser.add_argument('--lr_stage', type=float, default=0.001, help='Learning rate for optimize')
    parser.add_argument('--num_iter_stage', type=int, default=500, help='Number of iterations for optimize')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value')
    parser.add_argument('--attack_method', type=str, default='PGD_psy', help='Attack method')
    parser.add_argument('--save_path', type=str, default='adv_example')
    parser.add_argument('--alpha', type=float, default=0.07)
    args = parser.parse_args()
    main(args)
