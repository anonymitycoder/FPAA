import pystoi

from model import *
from dataset import *
from PGD_freq import *
from utils import *

s_criterion = nn.CrossEntropyLoss()
window_size = 2048
lr_stage2 = 0.1


def attack_stage(audios, model, labels, adv_distortion, th_batch, psd_max_batch, lr_stage2, nums_sample,
                 num_iter):
    delta = adv_distortion.clone().detach().requires_grad_(True)
    th_loss = torch.tensor([[np.inf] * nums_sample]).reshape((nums_sample, 1)).to(device)
    alpha = torch.ones((nums_sample, 1)) * 0.13
    alpha = alpha.to(device)
    final_alpha = torch.zeros(nums_sample)
    final_adv2 = torch.zeros_like(audios)
    optimizer2 = torch.optim.Adam([delta], lr=lr_stage2)
    min_th = -np.inf
    for i in range(num_iter):
        new_input = delta + audios

        new_input_stego = torch.clamp(new_input, 0, float('inf'))
        stego_output = model(new_input_stego)
        _, predicted = torch.max(stego_output, 1)
        cross_loss = s_criterion(stego_output, labels).to(device)
        th_loss_temp = compute_loss_th(delta, window_size, th_batch, psd_max_batch).to(device)
        total_loss = -cross_loss + (alpha * th_loss_temp).sum()

        optimizer2.zero_grad()
        total_loss.backward()
        optimizer2.step()

        th_loss_output = th_loss_temp.cpu().detach().numpy()
        alpha_output = alpha.cpu().detach().numpy()

        for ii in range(nums_sample):
            if predicted[ii] != labels[ii]:
                if th_loss_temp[ii] < th_loss[ii]:
                    th_loss[ii] = th_loss_temp[ii]
                    final_alpha[ii] = alpha[ii]
                    final_adv2[ii] = new_input[ii]
                    print('==============================Attack Succeed!==============================')
            if i % 20 == 0:
                alpha[ii] *= 1.2
            if i % 20 == 0 and predicted[ii] == labels[ii]:
                alpha[ii] *= 0.8
                alpha[ii] = max(alpha[ii], min_th)
            print('Iteration [{}/{}], th_loss: {}, '
                  'alpha: {}'.format(ii + 1, i + 1, th_loss_output[ii], alpha_output[ii]))
            if i == num_iter - 1 and (final_adv2[ii] == 0).all():
                final_adv2[ii] = new_input[ii]
              
    return final_adv2, th_loss, final_alpha


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_name, num_classes):
    model_mapping = {
        'VGG13': VGG13,
        'VGG16': VGG16,
        'GoogLeNet': GoogLeNet,
        'ResNet18': resnet18,
        'DenseNet': DenseNet,
        'SB_CNN': SB_CNN,
    }
    model_class = model_mapping.get(model_name)
    if model_class is None:
        raise ValueError(f"Model {model_name} not recognized.")

    if model_name == 'SB_CNN' and args.dataset == 'ESC-50':
        model = model_class(bands=216, num_labels=num_classes)
    elif model_name == 'SB_CNN' and args.dataset != 'ESC-50':
        model = model_class()
    else:
        model = model_class(num_classes=num_classes)

    model_path = f'model_weights/{model_name}.pth' if num_classes == 10 else f'model_weights/{model_name}_esc50.pth'
    model.load_state_dict(torch.load(model_path), strict=False)
    return model.to(device).eval()


def main(args):
    num_classes = 10 if args.dataset == 'UrbanSound8K' else 50
    nums_sample = 50 if args.dataset == 'UrbanSound8K' else 10

    model_names = ['VGG13','VGG16','DenseNet','GoogLeNet','SB_CNN']
    for model_name in model_names:
        model = load_model(model_name, num_classes=num_classes)
        if args.dataset == 'UrbanSound8K':
            psd_path = 'psd/UrbanSound8k_psd'
        else:
            psd_path = 'psdESC-50_psd'

        save_path = Path(args.save_path) / args.dataset / model_name / args.attack_method
        save_path.mkdir(parents=True, exist_ok=True)

        for class_id in tqdm(range(num_classes), desc='Processing classes'):
            print(f'\nThe current class under attack is the {class_id}th class.')

            test_data = torch.load(os.path.join('UrbanSound8K_data', f'class{class_id}', 'real', 'correct_real.pt'))
            test_labels = torch.load(
                os.path.join('UrbanSound8K_data', f'class{class_id}', 'label', 'correct_labels.pt'))

            adv_audio = torch.load(
                args.save_path + '/' + args.dataset + '/' + model_name + '/' + 'PGD' + '/' + f'class_{class_id}_adv_audio.pt')

            th_batch = torch.load(f'{psd_path}/class{class_id}/th_batch.pt')
            th_batch = th_batch.to(device)

            psd_max_batch = torch.load(f'{psd_path}/class{class_id}\\psd_max_batch.pt')
            # psd_max_batch = torch.stack(psd_max_batch)
            psd_max_batch = psd_max_batch.to(device)

            adv = adv_audio - test_data.clone()
            adv_example, th_loss, final_alpha = attack_stage(test_data, model, test_labels, adv, th_batch,
                                                             psd_max_batch, lr_stage2, nums_sample=nums_sample,
                                                             num_iter=1000)

            torch.save(adv_example, save_path / f'class_{class_id}_adv_audio.pt')


if __name__ == '__main__':
      parser = argparse.ArgumentParser(description='Adversarial attack on audio models')
      parser.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K', 'ESC-50'],
                            help='Dataset name')
      parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for the attack')
      parser.add_argument('--save_path', type=str, default='adv_examples', help='Path to save adversarial examples')
      parser.add_argument('--attack_method', type=str, default=f'FPAA', help='Attack method to use')
      args = parser.parse_args()
      main(args)
