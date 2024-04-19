import argparse
from pathlib import Path
import torch
from autoattack import AutoAttack
from autoattack.autopgd_base import APGDAttack
from torchvision.models import densenet121
import torch.nn as nn

from model import VGG13, VGG16, GoogLeNet, DenseNet, SB_CNN

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
    model_names = [
        'DenseNet']
    for model_name in model_names:
        print('#################################')
        print(f"Processing model: {model_name}")

        num_classes = 10 if args.dataset == 'UrbanSound8K' else 50
        data_path = Path('UrbanSound8K_data') if args.dataset == 'UrbanSound8K' else Path('ESC-50_data')

        model = load_model(model_name, num_classes=num_classes)

        # adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard')
        adversary = APGDAttack(model, n_restarts=1, n_iter=10, verbose=True,
                               eps=args.epsilon, norm='Linf', eot_iter=1, rho=.75,seed=1,loss='ce')

        save_path = Path(args.save_path) / args.dataset / model_name / args.attack_method
        save_path.mkdir(parents=True, exist_ok=True)

        for class_id in range(num_classes):
            print(f'The current class under attack is the {class_id}th class.')
            data = torch.load(data_path / f'class{class_id}' / 'real' / 'correct_real.pt')[0:25]
            label = torch.load(data_path / f'class{class_id}' / 'label' / 'correct_labels.pt')[0:25]

            x_adv = adversary.perturb(data, label)

            torch.save(x_adv, save_path / f'class_{class_id}_adv_audio.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial attack on audio models')
    parser.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K', 'ESC-50'],
                        help='Dataset name')
    parser.add_argument('--epsilon', type=float, default=0.0028, help='Epsilon value for the attack')
    parser.add_argument('--save_path', type=str, default='adv_example_02', help='Path to save adversarial examples')
    parser.add_argument('--attack_method', type=str, default='Autoattack', help='Attack method to use')
    args = parser.parse_args()
    main(args)
