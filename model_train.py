import argparse

import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UrbanSound8KDataset, Esc50Dataset
from model import *


def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for (inputs, labels) in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    return train_loss


def test_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and testing script for dataset')
    parser.add_argument('--dataset', type=str, default='Urban8K', help='Dataset name')
    parser.add_argument('--model', type=str, default='VGG13', help='Model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'Urban8K':
        train_dataset = UrbanSound8KDataset("dataset/UrbanSound8K", train=True)
        test_dataset = UrbanSound8KDataset("dataset/UrbanSound8K", train=False)
        if args.model == 'VGG13':
            model = VGG13(num_classes=10, init_weights=True).to(device)
        elif args.model == 'VGG16':
            model = VGG16(num_classes=10, init_weights=True).to(device)
        else:
            model = CRNN(num_classes=10, init_weights=True).to(device)
    else:
        train_dataset = Esc50Dataset("dataset/ESC-50", train=True)
        test_dataset = Esc50Dataset("dataset/ESC-50", train=False)
        if args.model == 'VGG13':
            model = VGG13(num_classes=50, init_weights=True).to(device)
        elif args.model == 'VGG16':
            model = VGG16(num_classes=50, init_weights=True).to(device)
        else:
            model = CRNN(num_classes=50, init_weights=True).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        acc = test_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, test_acc={acc:.4f}")
    model_state_dict = model.state_dict()
    if args.dataset == 'Urban8K':
        torch.save(model_state_dict, f"model_state/{args.model}.pth")
    else:
        torch.save(model_state_dict, f"model_state/{args.model}_esc50.pth")
