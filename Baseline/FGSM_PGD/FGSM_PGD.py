import argparse
from pathlib import Path
import torch
from foolbox import PyTorchModel
from foolbox.attacks import FGSM, LinfPGD
from model import VGG13, VGG16, GoogLeNet, resnet18, DenseNet, SB_CNN

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
    model_names = ['VGG13', 'VGG16', 'GoogLeNet', 'ResNet18', 'DenseNet',
                   'SB_CNN']  # Example model to process, expand as needed
    for model_name in model_names:
        print(f"Processing model: {model_name}")
        if args.attack_method == 'FGSM':
            attack = FGSM(random_start=True)
        elif args.attack_method == 'PGD':
            attack = LinfPGD(steps=40, rel_stepsize=1.0 / 40, random_start=True)
        else:
            raise ValueError("Unsupported attack method. Choose 'FGSM' or 'PGD'.")

        num_classes = 10 if args.dataset == 'UrbanSound8K' else 50
        data_path = Path('UrbanSound8K_data') if args.dataset == 'UrbanSound8K' else Path('ESC-50_data')

        model = load_model(model_name, num_classes=num_classes)
        f_model = PyTorchModel(model, bounds=(0, float('inf')))
        save_path = Path(args.save_path) / args.dataset / model_name / args.attack_method
        save_path.mkdir(parents=True, exist_ok=True)

        acc = []

        for class_id in range(num_classes):
            print(f'The current class under attack is the {class_id}th class.')
            data = torch.load(data_path / f'class{class_id}' / 'real' / 'correct_real.pt')
            label = torch.load(data_path / f'class{class_id}' / 'label' / 'correct_labels.pt')
            _, adv_example, success = attack(f_model, data, label, epsilons=args.epsilon)
            attack_success_rate = success.float().mean().item()
            print(f'Attack success rate: {attack_success_rate}')
            acc.append(attack_success_rate)

            torch.save(adv_example, save_path / f'class_{class_id}_adv_audio.pt')
        print(sum(acc) / len(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial attack on audio models')
    parser.add_argument('--dataset', type=str, default='ESC-50', choices=['UrbanSound8K', 'ESC-50'],
                        help='Dataset name')
    parser.add_argument('--epsilon', type=float, default=1.1, help='Epsilon value for the attack')
    parser.add_argument('--save_path', type=str, default='adv_example_02', help='Path to save adversarial examples')
    parser.add_argument('--attack_method', type=str, default='FGSM', choices=['FGSM', 'PGD'],
                        help='Attack method to use')
    args = parser.parse_args()
    main(args)
