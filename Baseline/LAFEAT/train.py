import argparse
import os
import time
from pathlib import Path
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from frozen import *
from dataset import UrbanSound8KDataset, Esc50Dataset
from model import VGG13, VGG16, resnet18, GoogLeNet, SB_CNN, DenseNet
import numpy as np
from utils import AverageMeter
import datetime
from pathlib import Path


def train(trainloader, model, criterion, optimizer, model_lr, print_freq):
    model.train()
    meters = [AverageMeter() for _ in range(1)]

    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.cuda(), labels.cuda()
        all_outputs = model(data)

        model_label = all_outputs[-1].max(1)[1].detach()

        loss_xent = criterion(all_outputs[0], model_label)

        optimizer.zero_grad()
        loss_xent.backward()
        optimizer.step()
        
        meters[0].update(loss_xent.item(), labels.size(0))

        if (batch_idx + 1) % print_freq == 0:
            print(f'Batch {batch_idx + 1}/{len(trainloader)}, Loss: {meters[0].avg:.3f}')

    model_lr.step()

def test(model, testloader):
    model.eval()
    total = 0
    corrects = torch.zeros(5, device='cuda')

    with torch.no_grad():
        for data, labels in testloader:
            data = data.cuda()
            labels = labels.cuda()
            outputs = model(data)[-5:]
            predictions = torch.stack([o.max(0)[1] if o.dim() == 1 else o.max(1)[1] for o in outputs])

            num_predictions = len(predictions)
            for i in range(num_predictions):
                corrects[i] += (predictions[i] == labels).sum()
            total += labels.size(0)

    accs = corrects.float() / total
    return accs.cpu().numpy(), total
    
def load_model(model_name):
    model_mapping = {
        'VGG13': VGG13,
        'VGG16': VGG16,
        'GoogLeNet': GoogLeNet,
        'ResNet18': resnet18,
        'DenseNet': DenseNet,
        'SB_CNN': SB_CNN,
    }
    model_class = model_mapping.get(model_name)
    return model_class


def load_datasets(args):
    dataset_path = "D:/UrbanSound8K" if args.dataset == 'UrbanSound8K' else "D:/ESC-50-master"
    if args.dataset == 'UrbanSound8K':
        return UrbanSound8KDataset(dataset_path, train=True), UrbanSound8KDataset(dataset_path, train=False)
    else:
        return Esc50Dataset(dataset_path, train=True), Esc50Dataset(dataset_path, train=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test audio models')
    parser.add_argument('--dataset', type=str, default='ESC-50', choices=['UrbanSound8K', 'ESC-50'],
                        help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--print-freq', type=int, default=5)
    parser.add_argument('--model_dir', type=str, default='models', help='Path to checkpoint')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--max-epoch', type=int, default=25)
    parser.add_argument('--eval-freq', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = load_datasets(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    torch.manual_seed(args.seed)

    model_names = ['VGG13','VGG16','DenseNet','GoogLeNet','SB_CNN']
    for model_name in model_names:
        print(f"Processing model: {model_name}")
        num_classes = 10 if args.dataset == 'UrbanSound8K' else 50
        model_class = load_model(model_name)
        if model_class is None:
            raise ValueError(f"Model {model_name} not recognized.")

        model = Frozen_VGG13_esc50(model_class, num_classes=num_classes)

        model_path = f'E:/PyCharm_Projects/FPAA/model_weights/{model_name}.pth' if num_classes == 10 else f'E:/PyCharm_Projects/FPAA/model_weights/{model_}_esc50.pth'
        model.load_frozen(model_path)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        model_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-5)
        start_time = time.time()
        for epoch in range(args.max_epoch):
            print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
            print('LR: %f' % (model_lr.get_last_lr()[-1]))
            train(
                train_loader, model, criterion,
                optimizer, model_lr, args.print_freq)
            dotest = (args.eval_freq > 0 and epoch % args.eval_freq == 0 or (epoch + 1) == args.max_epoch)
            if dotest:
                accs, total = test(model, test_loader)
                desc = ', '.join(f'Acc256_{i + 15}: {a:.2%}' for i, a in enumerate(accs[:-1]))
                print(f'{desc}, Acc_outputs: {accs[-1]:.2%}, Total: {total}')
            # save models
        checkpoint = {
            'epoch': args.max_epoch,
            'state_dict': model.state_dict(),
            'optimizer_model': optimizer.state_dict(), }
        model_dir_path = Path(args.model_dir)
        save_path = model_dir_path / f"{model_}_{'urban8k' if args.dataset == 'UrbanSound8K' else 'esc50'}.pt"
        torch.save(checkpoint, save_path)
        print(f'Saved logits models at {save_path}.')
        elapsed = round(time.time() - start_time)
        elapsed = datetime.timedelta(seconds=elapsed)
        print(f'Finished. Total elapsed time (h:m:s): {elapsed}.')
