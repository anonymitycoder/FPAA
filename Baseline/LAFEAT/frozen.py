import torch.nn as nn
import torch
import torch.nn.functional as F


class Frozen_VGG13_urban8k(nn.Module):
    def __init__(self, model_cls, num_classes=10):
        super().__init__()
        model = model_cls(num_classes=num_classes, latent=True)
        for p in model.parameters():
            p.requires_grad = False
        self.frozen_model = nn.DataParallel(model)

        self.extra_conv = nn.Conv2d(512, 1024, kernel_size=3, padding=1).cuda()
        self.extra_pool = nn.MaxPool2d(kernel_size=2, stride=2).cuda()

        self.extra_fc = nn.Linear(1024 * 3 * 3, num_classes).cuda()

    def load_frozen(self, filename):
        state = torch.load(filename)
        self.frozen_model.module.load_state_dict(state)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')
        }

    def forward(self, x):
        pooled_output, before_last_fc_output, final_output = self.frozen_model(x)
        x = self.extra_conv(pooled_output)
        x = F.relu(x)
        x = self.extra_pool(x)
        x = torch.flatten(x, 1)
        extra_fc_output = self.extra_fc(x)
        return extra_fc_output, final_output


class Frozen_VGG13_esc50(nn.Module):
    def __init__(self, model_cls, num_classes=10):
        super().__init__()
        model = model_cls(num_classes=num_classes, latent=True)
        for p in model.parameters():
            p.requires_grad = False
        self.frozen_model = nn.DataParallel(model)


        self.extra_conv = nn.Conv2d(512, 1024, kernel_size=3, padding=1).cuda()
        self.extra_pool = nn.MaxPool2d(kernel_size=2, stride=2).cuda()


        self.extra_fc = nn.Linear(1024 * 3 * 3, num_classes).cuda()

    def load_frozen(self, filename):
        state = torch.load(filename)
        self.frozen_model.module.load_state_dict(state)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')
        }

    def forward(self, x):
        pooled_output, before_last_fc_output, final_output = self.frozen_model(x)
        x = self.extra_conv(pooled_output)
        x = F.relu(x)
        x = self.extra_pool(x)
        x = torch.flatten(x, 1)
        extra_fc_output = self.extra_fc(x)
        return extra_fc_output, final_output


class Frozen_VGG16_urban8k(nn.Module):
    def __init__(self, model_cls, num_classes=10):
        super().__init__()
        model = model_cls(num_classes=num_classes, latent=True)
        for p in model.parameters():
            p.requires_grad = False
        self.frozen_model = nn.DataParallel(model)


        self.extra_conv = nn.Conv2d(512, 1024, kernel_size=3, padding=1).cuda()

        self.extra_pool = nn.MaxPool2d(kernel_size=2, stride=2).cuda()

        self.extra_fc = nn.Linear(1024 * 32, num_classes).cuda()

    def load_frozen(self, filename):
        state = torch.load(filename)
        self.frozen_model.module.load_state_dict(state)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')
        }

    def forward(self, x):
        pooled_output, before_last_fc_output, final_output = self.frozen_model(x)


        x = self.extra_conv(pooled_output)
        x = F.relu(x)
        x = self.extra_pool(x)

        x = torch.flatten(x, 1)

        extra_fc_output = self.extra_fc(x)
        return extra_fc_output, final_output


class Frozen_VGG16_esc50(nn.Module):
    def __init__(self, model_cls, num_classes=10):
        super().__init__()
        model = model_cls(num_classes=num_classes, latent=True)
        for p in model.parameters():
            p.requires_grad = False
        self.frozen_model = nn.DataParallel(model)


        self.extra_conv = nn.Conv2d(512, 1024, kernel_size=3, padding=1).cuda()

        self.extra_pool = nn.MaxPool2d(kernel_size=2, stride=2).cuda()

        self.extra_fc = nn.Linear(49152, num_classes).cuda()

    def load_frozen(self, filename):
        state = torch.load(filename)
        self.frozen_model.module.load_state_dict(state)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')
        }

    def forward(self, x):
        pooled_output, before_last_fc_output, final_output = self.frozen_model(x)

        x = self.extra_conv(pooled_output)
        x = F.relu(x)
        x = self.extra_pool(x)

        x = torch.flatten(x, 1)

        extra_fc_output = self.extra_fc(x)
        return extra_fc_output, final_output


class Frozen_ResNet_urban8k(nn.Module):
    def __init__(self, model_cls, num_classes=10):
        super().__init__()
        model = model_cls(num_classes=num_classes, latent=True)
        for p in model.parameters():
            p.requires_grad = False
        self.frozen_model = nn.DataParallel(model)

        self.extra_conv = nn.Conv2d(256, 1024, kernel_size=3, padding=1).cuda()
        self.extra_pool = nn.AvgPool2d(kernel_size=2, stride=2).cuda()

        self.extra_fc = nn.Linear(1024 * 160, num_classes).cuda()

    def load_frozen(self, filename):
        state = torch.load(filename)
        self.frozen_model.module.load_state_dict(state)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')
        }

    def forward(self, x):
        pooled_output, before_last_fc_output, final_output = self.frozen_model(x)

        x = self.extra_conv(pooled_output)
        x = F.relu(x)
        x = self.extra_pool(x)

        x = torch.flatten(x, 1)

        extra_fc_output = self.extra_fc(x)
        return extra_fc_output, final_output


class Frozen_ResNet_esc50(nn.Module):
    def __init__(self, model_cls, num_classes=10):
        super().__init__()
        model = model_cls(num_classes=num_classes, latent=True)
        for p in model.parameters():
            p.requires_grad = False
        self.frozen_model = nn.DataParallel(model)

        self.extra_conv = nn.Conv2d(256, 1024, kernel_size=3, padding=1).cuda()
        self.extra_pool = nn.AvgPool2d(kernel_size=2, stride=2).cuda()

        self.extra_fc = nn.Linear(229376, num_classes).cuda()

    def load_frozen(self, filename):
        state = torch.load(filename)
        self.frozen_model.module.load_state_dict(state)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')
        }

    def forward(self, x):
        pooled_output, before_last_fc_output, final_output = self.frozen_model(x)

        x = self.extra_conv(pooled_output)
        x = F.relu(x)
        x = self.extra_pool(x)

        x = torch.flatten(x, 1)

        extra_fc_output = self.extra_fc(x)
        return extra_fc_output, final_output


class Frozen_GoogLeNet_urban8k(nn.Module):
    def __init__(self, model_cls, num_classes=10):
        super().__init__()
        model = model_cls(num_classes=num_classes, latent=True)
        for p in model.parameters():
            p.requires_grad = False
        self.frozen_model = nn.DataParallel(model)

        self.extra_conv = nn.Conv2d(480, 1024, kernel_size=3, padding=1).cuda()

        self.extra_pool = nn.AvgPool2d(kernel_size=2, stride=2).cuda()

        self.extra_fc = nn.Linear(720896, num_classes).cuda()

    def load_frozen(self, filename):
        state = torch.load(filename)
        self.frozen_model.module.load_state_dict(state)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')
        }

    def forward(self, x):
        pooled_output, before_last_fc_output, final_output = self.frozen_model(x)

        x = self.extra_conv(pooled_output)
        x = F.relu(x)
        x = self.extra_pool(x)

        x = torch.flatten(x, 1)

        extra_fc_output = self.extra_fc(x)
        return extra_fc_output, final_output


class Frozen_GoogLeNet_esc50(nn.Module):
    def __init__(self, model_cls, num_classes=10):
        super().__init__()
        model = model_cls(num_classes=num_classes, latent=True)
        for p in model.parameters():
            p.requires_grad = False
        self.frozen_model = nn.DataParallel(model)

        self.extra_conv = nn.Conv2d(480, 1024, kernel_size=3, padding=1).cuda()

        self.extra_pool = nn.AvgPool2d(kernel_size=2, stride=2).cuda()

        self.extra_fc = nn.Linear(851968, num_classes).cuda()

    def load_frozen(self, filename):
        state = torch.load(filename)
        self.frozen_model.module.load_state_dict(state)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')
        }

    def forward(self, x):
        pooled_output, before_last_fc_output, final_output = self.frozen_model(x)

        x = self.extra_conv(pooled_output)
        x = F.relu(x)
        x = self.extra_pool(x)

        x = torch.flatten(x, 1)

        extra_fc_output = self.extra_fc(x)
        return extra_fc_output, final_output


class Frozen_SB_CNN_urban8k(nn.Module):
    def __init__(self, model_cls, frames=1025, bands=173, num_classes=10):
        super().__init__()
        model = model_cls(frames=frames, bands=bands, num_labels=num_classes, latent=True)
        for p in model.parameters():
            p.requires_grad = False
        self.frozen_model = nn.DataParallel(model)

        self.extra_conv = nn.Conv2d(48, 1024, kernel_size=3, padding=1).cuda()

        self.extra_pool = nn.AvgPool2d(kernel_size=2, stride=2).cuda()

        self.extra_fc = nn.Linear(534528, num_classes).cuda()

    def load_frozen(self, filename):
        state = torch.load(filename)
        self.frozen_model.module.load_state_dict(state)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')
        }

    def forward(self, x):
        pooled_output, before_last_fc_output, final_output = self.frozen_model(x)

        x = self.extra_conv(pooled_output)
        x = F.relu(x)
        x = self.extra_pool(x)

        x = torch.flatten(x, 1)

        extra_fc_output = self.extra_fc(x)
        return extra_fc_output, final_output


class Frozen_SB_CNN_esc50(nn.Module):
    def __init__(self, model_cls, frames=1025, bands=216, num_classes=50):
        super().__init__()
        model = model_cls(frames=frames, bands=bands, num_labels=num_classes, latent=True)
        for p in model.parameters():
            p.requires_grad = False
        self.frozen_model = nn.DataParallel(model)

        self.extra_conv = nn.Conv2d(48, 1024, kernel_size=3, padding=1).cuda()

        self.extra_pool = nn.AvgPool2d(kernel_size=2, stride=2).cuda()

        self.extra_fc = nn.Linear(683008, num_classes).cuda()

    def load_frozen(self, filename):
        state = torch.load(filename)
        self.frozen_model.module.load_state_dict(state)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')
        }

    def forward(self, x):
        pooled_output, before_last_fc_output, final_output = self.frozen_model(x)

        x = self.extra_conv(pooled_output)
        x = F.relu(x)
        x = self.extra_pool(x)

        x = torch.flatten(x, 1)

        extra_fc_output = self.extra_fc(x)
        return extra_fc_output, final_output


class Frozen_DenseNet_urban8k(nn.Module):
    def __init__(self, model_cls, num_classes=10):
        super().__init__()
        model = model_cls(num_classes=num_classes,latent=True)
        for p in model.parameters():
            p.requires_grad = False
        self.frozen_model = nn.DataParallel(model)

        self.extra_conv = nn.Conv2d(1024, 1024, kernel_size=3, padding=1).cuda()
        self.extra_pool = nn.MaxPool2d(kernel_size=2, stride=2).cuda()

        self.extra_fc = nn.Linear(32768, num_classes).cuda()

    def load_frozen(self, filename):
        state = torch.load(filename)
        self.frozen_model.module.load_state_dict(state, strict=False)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')
        }

    def forward(self, x):
        pooled_output, before_last_fc_output, final_output = self.frozen_model(x)
        x = self.extra_conv(pooled_output)
        x = F.relu(x)
        x = self.extra_pool(x)
        x = torch.flatten(x, 1)
        extra_fc_output = self.extra_fc(x)
        return extra_fc_output, final_output


class Frozen_DenseNet_esc50(nn.Module):
    def __init__(self, model_cls, num_classes=10):
        super().__init__()
        model = model_cls(num_classes=num_classes, latent=True)
        for p in model.parameters():
            p.requires_grad = False
        self.frozen_model = nn.DataParallel(model)

        self.extra_conv = nn.Conv2d(1024, 1024, kernel_size=3, padding=1).cuda()
        self.extra_pool = nn.MaxPool2d(kernel_size=2, stride=2).cuda()
        self.extra_fc = nn.Linear(49152, num_classes).cuda()

    def load_frozen(self, filename):
        state = torch.load(filename)
        self.frozen_model.module.load_state_dict(state, strict=False)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state.items()
            if not k.startswith('frozen_model.')
        }

    def forward(self, x):
        pooled_output, before_last_fc_output, final_output = self.frozen_model(x)
        x = self.extra_conv(pooled_output)
        x = F.relu(x)
        x = self.extra_pool(x)
        x = torch.flatten(x, 1)
        extra_fc_output = self.extra_fc(x)
        return extra_fc_output, final_output
