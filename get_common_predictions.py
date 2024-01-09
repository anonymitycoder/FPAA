import os
import random

from torch.utils.data import DataLoader

from dataset import ESC50OneClass
from model import *


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    base_path = 'esc_data'

    def filter_correct_predictions(class_, data_loader, model1, model2, model3, num_samples=10):
        correct_real = []
        correct_imaginary = []
        correct_labels = []
        for batch, (stft_db_magnitude, stft_db_phase, labels) in enumerate(data_loader):
            stft_db_magnitude = stft_db_magnitude.to(device)
            stft_db_phase = stft_db_phase.to(device)
            labels = labels.to(device)
            outputs1 = model1(stft_db_magnitude)
            outputs2 = model2(stft_db_magnitude)
            outputs3 = model3(stft_db_magnitude)
            predictions1 = torch.argmax(outputs1, dim=1)
            predictions2 = torch.argmax(outputs2, dim=1)
            predictions3 = torch.argmax(outputs3, dim=1)
            correct_mask = (predictions1 == labels) & (predictions2 == labels) & (predictions3 == labels)
            correct_real.extend(stft_db_magnitude[correct_mask])
            correct_imaginary.extend(stft_db_phase[correct_mask])
            correct_labels.extend(labels[correct_mask])

        random_indices = random.sample(range(len(correct_real)), min(num_samples, len(correct_real)))
        correct_real = torch.stack([correct_real[i] for i in random_indices])
        correct_imaginary = torch.stack([correct_imaginary[i] for i in random_indices])
        correct_labels = torch.stack([correct_labels[i] for i in random_indices])

        directory1 = os.path.join(base_path, f'class{class_}', 'real')
        ensure_directory_exists(directory1)
        directory2 = os.path.join(base_path, f'class{class_}', 'imaginary')
        ensure_directory_exists(directory2)
        directory3 = os.path.join(base_path, f'class{class_}', 'label')
        ensure_directory_exists(directory3)

        torch.save(correct_real, os.path.join(directory1, 'correct_real.pt'))
        torch.save(correct_imaginary, os.path.join(directory2, 'correct_imaginary.pt'))
        torch.save(correct_labels, os.path.join(directory3, 'correct_labels.pt'))

    model1 = VGG13(num_classes=50)
    model1.load_state_dict(torch.load('model_state/VGG13_esc50.pth'))
    model1.eval()
    model2 = VGG16(num_classes=50)
    model2.load_state_dict(torch.load('model_state/VGG16_esc50.pth'))
    model2.eval()
    model3 = CRNN(num_classes=50)
    model3.load_state_dict(torch.load('model_state/CRNN_esc50.pth'))
    model3.eval()

    for i in range(50):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model1.to(device)
        model2.to(device)
        model3.to(device)
        datasets = ESC50OneClass("dataset\ESC-50", class_num=i)
        data_loader = DataLoader(datasets, batch_size=6, shuffle=False)
        filter_correct_predictions(i, data_loader, model1, model2, model3)


if __name__ == "__main__":
    main()
