import time
import math

import numpy as np
import torch
from torch import nn


class LafeatAttack():
    def __init__(
            self, model, n_iter=100, norm='Linf', eps=None,
            loss='ce', eot_iter=1, rho=.75,
            verbose=False, device='cuda', seed=0):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps - 1.9 * 1e-8
        self.norm = norm
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.device = device
        self.scale = True
        self.linear = True

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (
                x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def check_right_index(self, output, labels):
        output_index = output.argmax(dim=-1) == labels
        mask = output_index.to(dtype=torch.int8)
        mask = torch.unsqueeze(mask, -1)
        return mask

    def get_output_scale(self, output):
        std_max_out = []
        maxk = max((10,))
        pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)
        std_max_out.extend((pred_val_out[:, 0] - pred_val_out[:, 1] + 1e-8).cpu().numpy())
        scale_list = [item / 1.0 for item in std_max_out]
        scale_list = torch.tensor(scale_list).to(self.device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (
                t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (
                    (t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(0., 1.)
        x_best_adv = x_adv.clone()

        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduce=False, reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknown loss')

        step_size_begin = self.eps * 2.0 / self.n_iter  # Initial step size
        x_adv_old = x_adv.clone()

        for i in range(self.n_iter):
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)

            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    outputs = self.model(x_adv)
                    out_adv = outputs[-1]  # Use the final output for calculating the loss
                    loss = criterion_indiv(out_adv, y)
                    loss = loss.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)
            with torch.no_grad():
                step_size = step_size_begin * (1 + math.cos(i / self.n_iter * math.pi)) * 0.5
                if self.norm == 'Linf':
                    x_adv = x_adv + step_size * torch.sign(grad)
                    x_adv = torch.min(torch.max(x_adv, x - self.eps), x + self.eps)  # Clip to epsilon constraint
                elif self.norm == 'L2':
                    g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                    scaled_grad = grad / (g_norm + 1e-10)
                    x_adv = x_adv + step_size * scaled_grad
                    diff = x_adv - x
                    diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
                    x_adv = x + diff
                x_adv = torch.clamp(x_adv, 0.0, 1.0)  # Clip to valid pixel range

            # Evaluate the success of the current adversarial examples
            logits = self.model(x_adv)[-1]
            pred = logits.max(1)[1] == y
            acc = torch.min(pred, pred)  # Update the accuracy

            # Update the best adversarial examples found so far
            pi = (pred == 0).nonzero(as_tuple=False).squeeze()
            if pi.numel() > 0:
                x_best_adv[pi] = x_adv[pi].clone()

        return acc, x_best_adv

    def perturb(self, x_in, y_in, scale_17=0):
        self.scale_17 = scale_17
        # self.seed = np.int(scale_17*10)
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        adv = x.clone()
        x_input = x



        acc = self.model(x_input)[-1].max(1)[1] == y
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(
                self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        ind_to_fool = acc.nonzero(as_tuple=False).squeeze()
        if len(ind_to_fool.shape) == 0:
            ind_to_fool = ind_to_fool.unsqueeze(0)
        if ind_to_fool.numel() != 0:
            x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
            acc_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
            ind_curr = (acc_curr == 0).nonzero(as_tuple=False).squeeze()
            #
            acc[ind_to_fool[ind_curr]] = 0
            adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
            if self.verbose:
                print(
                    f'robust accuracy: {acc.float().mean():.2%} - '
                    f'cum. time: {time.time() - startt:.1f}s')

        return acc, adv