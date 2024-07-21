'''
FAKEBOB attack was proposed in the paper "Who is real Bob? Adversarial Attacks on Speaker Recognition Systems"
accepted by the conference IEEE S&P (Oakland) 2021.
'''
import argparse
import warnings
from collections import Counter
from abc import ABCMeta, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from pystoi import stoi
from torch.utils.data import DataLoader

from model import ESC_Net

from defended_model import defended_model
from model import CNN
from train_ESC50 import DataGenerator


class Attack(metaclass=ABCMeta):

    @abstractmethod
    def attack(self, x, y, verbose=1, EOT_size=1, EOT_batch_size=1):
        pass

    def compare(self, y, y_pred, targeted):
        if targeted:
            return (y_pred == y).tolist()
        else:
            return (y_pred != y).tolist()


class SEC4SR_CrossEntropy(nn.CrossEntropyLoss):  # deal with something special on top of CrossEntropyLoss

    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', task='CSI'):
        super().__init__(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce,
                         reduction=reduction)

        assert task == 'CSI'  # CrossEntropy only supports CSI task

    def forward(self, scores, label):

        _, num_class = scores.shape
        device = scores.device
        label = label.to(device)
        loss = torch.zeros(label.shape[0], dtype=torch.float, device=scores.device)

        consider_index = torch.nonzero(label != -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
        if len(consider_index) > 0:
            loss[consider_index] = super().forward(scores[consider_index], label[consider_index])

        imposter_index = torch.nonzero(label == -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
        if len(imposter_index):
            loss[imposter_index] = 0. * torch.sum(scores[imposter_index])  # make backward

        return loss


class SEC4SR_MarginLoss(nn.Module):  # deal with something special on top of MarginLoss

    def __init__(self, targeted=False, confidence=0., task='CSI', threshold=None, clip_max=True) -> None:
        super().__init__()
        self.targeted = targeted
        self.confidence = confidence
        self.task = task
        self.threshold = threshold
        self.clip_max = clip_max

    def forward(self, scores, label):
        _, num_class = scores.shape
        device = scores.device
        label = label.to(device)
        loss = torch.zeros(label.shape[0], dtype=torch.float, device=scores.device)
        confidence = torch.tensor(self.confidence, dtype=torch.float, device=device)

        if self.task == 'SV':
            enroll_index = torch.nonzero(label == 0, as_tuple=True)[0].detach().cpu().numpy().tolist()
            imposter_index = torch.nonzero(label == -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
            assert len(enroll_index) + len(imposter_index) == label.shape[
                0], 'SV task should not have labels out of 0 and -1'
            if len(enroll_index) > 0:
                if self.targeted:
                    loss[enroll_index] = self.threshold + confidence - scores[enroll_index].squeeze(
                        1)  # imposter --> enroll, authentication bypass
                else:
                    loss[enroll_index] = scores[enroll_index].squeeze(
                        1) + confidence - self.threshold  # enroll --> imposter, Denial of Service
            if len(imposter_index) > 0:
                if self.targeted:
                    loss[imposter_index] = scores[imposter_index].squeeze(
                        1) + confidence - self.threshold  # enroll --> imposter, Denial of Service
                else:
                    loss[imposter_index] = self.threshold + confidence - scores[imposter_index].squeeze(
                        1)  # imposter --> enroll, authentication bypass

        elif self.task == 'CSI' or self.task == 'OSI':
            # remove imposter index which is unmeaningful for CSI task
            consider_index = torch.nonzero(label != -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
            if len(consider_index) > 0:
                label_one_hot = torch.zeros((len(consider_index), num_class), dtype=torch.float, device=device)
                for i, ii in enumerate(consider_index):
                    index = int(label[ii])
                    label_one_hot[i][index] = 1
                score_real = torch.sum(label_one_hot * scores[consider_index], dim=1)
                score_other = torch.max((1 - label_one_hot) * scores[consider_index] - label_one_hot * 10000, dim=1)[0]
                if self.targeted:
                    loss[consider_index] = score_other + confidence - score_real if self.task == 'CSI' \
                        else torch.clamp(score_other, min=self.threshold) + confidence - score_real
                else:
                    if self.task == 'CSI':
                        loss[consider_index] = score_real + confidence - score_other
                    else:
                        f_reject = torch.max(scores[consider_index], 1)[
                                       0] + confidence - self.threshold  # spk m --> reject
                        f_mis = torch.clamp(score_real,
                                            min=self.threshold) + confidence - score_other  # spk_m --> spk_n
                        loss[consider_index] = torch.minimum(f_reject, f_mis)

            imposter_index = torch.nonzero(label == -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
            if self.task == 'OSI':
                # imposter_index = torch.nonzero(label == -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
                if len(imposter_index) > 0:
                    if self.targeted:
                        loss[imposter_index] = torch.max(scores[imposter_index], 1)[0] + confidence - self.threshold
                    else:
                        loss[imposter_index] = self.threshold + confidence - torch.max(scores[imposter_index], 1)[0]
            else:  # CSI
                if len(imposter_index):
                    loss[imposter_index] = 0. * torch.sum(scores[imposter_index])  # make backward

            # else:
            #     loss[imposter_index] = torch.zeros(len(imposter_index))

        if self.clip_max:
            loss = torch.max(torch.tensor(0, dtype=torch.float, device=device), loss)

        return loss


def resolve_loss(loss_name='Entropy', targeted=False, confidence=0., task='CSI', threshold=None, clip_max=True):
    assert loss_name in ['Entropy', 'Margin']
    assert task in ['CSI', 'SV', 'OSI']
    if task == 'SV' or task == 'OSI' or loss_name == 'Margin':  # SV/OSI: ignore loss name, force using Margin Loss
        loss = SEC4SR_MarginLoss(targeted=targeted, confidence=confidence, task=task, threshold=threshold,
                                 clip_max=clip_max)
        if (task == 'SV' or task == 'OSI') and loss_name == 'Entropy':
            warnings.warn('You are targeting {} task. Force using Margin Loss.')
    else:
        # loss = nn.CrossEntropyLoss(reduction='none') # ONLY FOR CSI TASK
        loss = SEC4SR_CrossEntropy(reduction='none', task='CSI')  # ONLY FOR CSI TASK
    grad_sign = (1 - 2 * int(targeted)) if loss_name == 'Entropy' else -1

    return loss, grad_sign


def resolve_prediction(decisions):
    # print(decisions)
    predict = []
    for d in decisions:
        counts = Counter(d)
        predict.append(counts.most_common(1)[0][0])
    predict = np.array(predict)
    return predict


class NES(nn.Module):

    def __init__(self, samples_per_draw, samples_per_draw_batch, sigma, EOT_wrapper):
        super().__init__()
        self.samples_per_draw = samples_per_draw
        self.samples_per_draw_batch_size = samples_per_draw_batch
        self.sigma = sigma
        self.EOT_wrapper = EOT_wrapper  # EOT wraps the model

    def forward(self, x, y):
        n_audios, n_channels, N = x.shape
        num_batches = self.samples_per_draw // self.samples_per_draw_batch_size
        for i in range(num_batches):
            noise = torch.randn([n_audios, self.samples_per_draw_batch_size // 2, n_channels, N], device=x.device)
            # noise = (torch.rand([n_audios, self.samples_per_draw_batch_size // 2, n_channels, N], device=x.device) * 2 - 1).sign()
            noise = torch.cat((noise, -noise), 1)
            if i == 0:
                noise = torch.cat((torch.zeros_like(x, device=x.device).unsqueeze(1), noise), 1)
            eval_input = noise * self.sigma + x.unsqueeze(1)
            eval_input = eval_input.view(-1, n_channels, N)  # (n_audios*samples_per_draw_batch_size, n_channels, N)
            eval_y = None
            for jj, y_ in enumerate(y):
                tmp = torch.tensor(
                    [y_] * (self.samples_per_draw_batch_size + 1 if i == 0 else self.samples_per_draw_batch_size),
                    dtype=torch.long, device=x.device)
                if jj == 0:
                    eval_y = tmp
                else:
                    eval_y = torch.cat((eval_y, tmp))
            # scores, loss, _ = EOT_wrapper(eval_input, eval_y, EOT_num_batches, self.EOT_batch_size, use_grad=False)
            scores, loss, _, decisions = self.EOT_wrapper(eval_input, eval_y)
            EOT_num_batches = int(self.EOT_wrapper.EOT_size // self.EOT_wrapper.EOT_batch_size)
            loss.data /= EOT_num_batches  # (n_audios*samples_per_draw_batch_size,)
            scores.data /= EOT_num_batches

            loss = loss.view(n_audios, -1)
            scores = scores.view(n_audios, -1, scores.shape[1])

            if i == 0:
                adver_loss = loss[..., 0]  # (n_audios, )
                loss = loss[..., 1:]  # (n_audios, samples_batch)
                adver_score = scores[:, 0, :]  # (n_audios, n_spks)
                noise = noise[:, 1:, :, :]  # (n_audios, samples_batch, n_channels, N)
                grad = torch.mean(loss.unsqueeze(2).unsqueeze(3) * noise, 1)
                mean_loss = loss.mean(1)
                predicts = resolve_prediction(decisions).reshape(n_audios, -1)  # (n_audios, samples_batch)
                predict = predicts[:, 0]
            else:
                grad += torch.mean(loss.unsqueeze(2).unsqueeze(3) * noise, 1)
                mean_loss += loss.mean(1)
        grad = grad / self.sigma / num_batches
        mean_loss = mean_loss / num_batches
        return mean_loss, grad, adver_loss, adver_score, predict


class EOT(nn.Module):

    def __init__(self, model, loss, EOT_size=1, EOT_batch_size=1, use_grad=True):
        super().__init__()
        self.model = model
        self.loss = loss
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.EOT_num_batches = self.EOT_size // self.EOT_batch_size
        self.use_grad = use_grad

    def forward(self, x_batch, y_batch, EOT_num_batches=None, EOT_batch_size=None, use_grad=None):
        EOT_num_batches = EOT_num_batches if EOT_num_batches else self.EOT_num_batches
        EOT_batch_size = EOT_batch_size if EOT_batch_size else self.EOT_batch_size
        use_grad = use_grad if use_grad else self.use_grad
        n_audios, n_channels, max_len = x_batch.size()
        grad = None
        scores = None
        loss = 0
        # decisions = [[]] * n_audios ## wrong, all element shares the same memory
        decisions = [[] for _ in range(n_audios)]
        for EOT_index in range(EOT_num_batches):
            x_batch_repeat = x_batch.repeat(EOT_batch_size, 1, 1)
            if use_grad:
                x_batch_repeat.retain_grad()
            y_batch_repeat = y_batch.repeat(EOT_batch_size)
            # scores_EOT = self.model(x_batch_repeat) # scores or logits. Just Name it scores. (batch_size, n_spks)
            decisions_EOT, scores_EOT = self.model.make_decision(
                x_batch_repeat)  # scores or logits. Just Name it scores. (batch_size, n_spks)
            loss_EOT = self.loss(scores_EOT, y_batch_repeat)
            if use_grad:
                loss_EOT.backward(torch.ones_like(loss_EOT))

            if EOT_index == 0:
                scores = scores_EOT.view(EOT_batch_size, -1, scores_EOT.shape[1]).mean(0)
                loss = loss_EOT.view(EOT_batch_size, -1).mean(0)
                if use_grad:
                    grad = x_batch_repeat.grad.view(EOT_batch_size, -1, n_channels, max_len).mean(0)
                    x_batch_repeat.grad.zero_()
            else:
                scores.data += scores_EOT.view(EOT_batch_size, -1, scores.shape[1]).mean(0)
                loss.data += loss_EOT.view(EOT_batch_size, -1).mean(0)
                if use_grad:
                    grad.data += x_batch_repeat.grad.view(EOT_batch_size, -1, n_channels, max_len).mean(0)
                    x_batch_repeat.grad.zero_()

            decisions_EOT = decisions_EOT.view(EOT_batch_size, -1).detach().cpu().numpy()
            for ii in range(n_audios):
                decisions[ii] += list(decisions_EOT[:, ii])

        return scores, loss, grad, decisions


class FAKEBOB(Attack):

    def __init__(self, model, threshold=None,
                 task='CSI', targeted=False, confidence=0.,
                 epsilon=0.002, max_iter=1000,
                 max_lr=0.001, min_lr=1e-6,
                 samples_per_draw=50, samples_per_draw_batch_size=50, sigma=0.001, momentum=0.9,
                 plateau_length=5, plateau_drop=2.,
                 stop_early=True, stop_early_iter=100,
                 batch_size=1, EOT_size=1, EOT_batch_size=1, verbose=1):

        self.model = model
        self.threshold = threshold
        self.task = task
        self.targeted = targeted
        self.confidence = confidence
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.samples_per_draw = samples_per_draw
        self.samples_per_draw_batch_size = samples_per_draw_batch_size
        self.sigma = sigma
        self.momentum = momentum
        self.plateau_length = plateau_length
        self.plateau_drop = plateau_drop
        self.stop_early = stop_early
        self.stop_early_iter = stop_early_iter
        self.batch_size = batch_size
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.verbose = verbose

        # loss_name = 'Margin'
        # self.loss, self.grad_sign = resolve_loss(loss_name, targeted, clip_max=False)

    def attack_batch(self, x_batch, y_batch, lower, upper, batch_id):

        with torch.no_grad():

            n_audios, _, _ = x_batch.shape

            last_ls = [[]] * n_audios
            lr = [self.max_lr] * n_audios
            prev_loss = [np.infty] * n_audios

            adver_x = x_batch.clone()
            grad = torch.zeros_like(x_batch, dtype=x_batch.dtype, device=x_batch.device)

            best_adver_x = adver_x.clone()
            best_loss = [np.infty] * n_audios
            consider_index = list(range(n_audios))

            for iter in range(self.max_iter + 1):
                prev_grad = grad.clone()
                # loss, grad, adver_loss, scores = self.get_grad(adver_x, y_batch)
                loss, grad, adver_loss, _, y_pred = self.get_grad(adver_x, y_batch)
                # y_pred = torch.max(scores, 1)[1].cpu().numpy()

                for ii, adver_l in enumerate(adver_loss):
                    index = consider_index[ii]
                    if adver_l < best_loss[index]:
                        best_loss[index] = adver_l.cpu().item()
                        best_adver_x[index] = adver_x[ii]

                if self.verbose:
                    print("batch: {} iter: {}, loss: {}, y: {}, y_pred: {}, best loss: {}".format(
                        batch_id, iter,
                        adver_loss.cpu().numpy(), y_batch.cpu().numpy(), y_pred, best_loss))

                # delete alrady found examples
                adver_x, y_batch, prev_grad, grad, lower, upper, \
                    consider_index, \
                    last_ls, lr, prev_loss, loss = self.delete_found(adver_loss, adver_x, y_batch, prev_grad, grad,
                                                                     lower, upper,
                                                                     consider_index, last_ls, lr, prev_loss, loss)
                if adver_x is None:  # all found
                    break

                if iter < self.max_iter:
                    grad = self.momentum * prev_grad + (1.0 - self.momentum) * grad
                    for jj, loss_ in enumerate(loss):
                        last_ls[jj].append(loss_)
                        last_ls[jj] = last_ls[jj][-self.plateau_length:]
                        if last_ls[jj][-1] > last_ls[jj][0] and len(last_ls[jj]) == self.plateau_length:
                            if lr[jj] > self.min_lr:
                                lr[jj] = max(lr[jj] / self.plateau_drop, self.min_lr)
                            last_ls[jj] = []

                    lr_t = torch.tensor(lr, device=adver_x.device, dtype=torch.float).unsqueeze(1).unsqueeze(2)
                    adver_x.data = adver_x + self.grad_sign * lr_t * torch.sign(grad)
                    adver_x.data = torch.min(torch.max(adver_x.data, lower), upper)

                    if self.stop_early and iter % self.stop_early_iter == 0:
                        loss_np = np.array([l.cpu() for l in loss])
                        converge_loss = np.array(prev_loss) * 0.9999 - loss_np
                        adver_x, y_batch, prev_grad, grad, lower, upper, \
                            consider_index, \
                            last_ls, lr, prev_loss, loss = self.delete_found(converge_loss, adver_x, y_batch, prev_grad,
                                                                             grad, lower, upper,
                                                                             consider_index, last_ls, lr, prev_loss,
                                                                             loss)
                        if adver_x is None:  # all converage
                            break

                        prev_loss = loss_np

            success = [False] * n_audios
            for kk, best_l in enumerate(best_loss):
                if best_l < 0:
                    success[kk] = True

            return best_adver_x, success

    def delete_found(self, adver_loss, adver_x, y_batch, prev_grad, grad, lower, upper,
                     consider_index, last_ls, lr, prev_loss, loss):
        adver_x_u = None
        y_batch_u = None
        prev_grad_u = None
        grad_u = None
        lower_u = None
        upper_u = None

        consider_index_u = []
        last_ls_u = []
        lr_u = []
        prev_loss_u = []
        loss_u = []

        for ii, adver_l in enumerate(adver_loss):
            if adver_l < 0:
                pass
            else:
                if adver_x_u is None:
                    adver_x_u = adver_x[ii:ii + 1, ...]
                    y_batch_u = y_batch[ii:ii + 1]
                    prev_grad_u = prev_grad[ii:ii + 1, ...]
                    grad_u = grad[ii:ii + 1, ...]
                    lower_u = lower[ii:ii + 1, ...]
                    upper_u = upper[ii:ii + 1, ...]
                else:
                    adver_x_u = torch.cat((adver_x_u, adver_x[ii:ii + 1, ...]), 0)
                    y_batch_u = torch.cat((y_batch_u, y_batch[ii:ii + 1]))
                    prev_grad_u = torch.cat((prev_grad_u, prev_grad[ii:ii + 1, ...]), 0)
                    grad_u = torch.cat((grad_u, grad[ii:ii + 1, ...]), 0)
                    lower_u = torch.cat((lower_u, lower[ii:ii + 1, ...]), 0)
                    upper_u = torch.cat((upper_u, upper[ii:ii + 1, ...]), 0)
                index = consider_index[ii]
                consider_index_u.append(index)
                last_ls_u.append(last_ls[ii])
                lr_u.append(lr[ii])
                prev_loss_u.append(prev_loss[ii])
                loss_u.append(loss[ii])

        return adver_x_u, y_batch_u, prev_grad_u, \
            grad_u, lower_u, upper_u, \
            consider_index_u, \
            last_ls_u, lr_u, prev_loss_u, loss_u

    def get_grad(self, x, y):
        NES_wrapper = NES(self.samples_per_draw, self.samples_per_draw_batch_size, self.sigma, self.EOT_wrapper)
        mean_loss, grad, adver_loss, adver_score, predict = NES_wrapper(x, y)

        return mean_loss, grad, adver_loss, adver_score, predict

    def attack(self, x, y):

        if self.task in ['SV', 'OSI'] and self.threshold is None:
            raise NotImplementedError('You are running black box attack for {} task, \
                        but the threshold not specified. Consider calling estimate threshold')
        self.loss, self.grad_sign = resolve_loss('Margin', self.targeted, self.confidence, self.task, self.threshold,
                                                 False)
        self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, False)

        lower = -2
        upper = 2
        assert lower <= x.max() <= upper, 'generating adversarial examples should be done in [-1, 1) float domain'
        n_audios, n_channels, _ = x.size()
        assert n_channels == 1, 'Only Support Mono Audio'
        assert y.shape[0] == n_audios, 'The number of x and y should be equal'
        upper = torch.clamp(x + self.epsilon, max=upper)
        lower = torch.clamp(x - self.epsilon, min=lower)

        batch_size = min(self.batch_size, n_audios)
        n_batches = int(np.ceil(n_audios / float(batch_size)))
        for batch_id in range(n_batches):
            x_batch = x[batch_id * batch_size:(batch_id + 1) * batch_size]  # (batch_size, 1, max_len)
            y_batch = y[batch_id * batch_size:(batch_id + 1) * batch_size]
            lower_batch = lower[batch_id * batch_size:(batch_id + 1) * batch_size]
            upper_batch = upper[batch_id * batch_size:(batch_id + 1) * batch_size]
            adver_x_batch, success_batch = self.attack_batch(x_batch, y_batch, lower_batch, upper_batch, batch_id)
            if batch_id == 0:
                adver_x = adver_x_batch
                success = success_batch
            else:
                adver_x = torch.cat((adver_x, adver_x_batch), 0)
                success += success_batch

        return adver_x, success

    def estimate_threshold_run(self, x, step=0.1):

        n_audios, _, _ = x.shape

        d, s = self.model.make_decision(x)
        d = d[0]
        s = s[0]
        if d != -1:
            return  # aleady accept, cannot be used to estimate threshold
        y = torch.tensor([-1] * n_audios, dtype=torch.long, device=x.device)
        init_score = np.max(s.cpu().numpy())
        delta = np.abs(init_score * step)
        threshold = init_score + delta

        adver_x = x.clone()
        grad = torch.zeros_like(x, dtype=x.dtype, device=x.device)

        lower = -1
        upper = 1
        upper = torch.clamp(x + self.epsilon, max=upper)
        lower = torch.clamp(x - self.epsilon, min=lower)

        iter_outer = 0
        n_iters = 0

        while True:
            self.loss, self.grad_sign = resolve_loss('Margin', False, 0., self.task, threshold, False)
            self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, False)

            iter_inner = 0

            last_ls = [[]] * n_audios
            lr = [self.max_lr] * n_audios

            while True:

                # test whether succeed
                decision, score = self.model.make_decision(adver_x)
                decision = decision[0]
                score = score[0]
                score = np.max(score.cpu().numpy())
                print(iter_outer, iter_inner, score, self.model.threshold)
                if decision != -1:  # succeed, found the threshold
                    return score
                elif score >= threshold:  # exceed the candidate threshold, but not succeed, exit the inner loop and increase the threshold
                    break

                # not succeed, update
                prev_grad = grad.clone()
                loss, grad, _, _, _ = self.get_grad(adver_x, y)

                grad = self.momentum * prev_grad + (1.0 - self.momentum) * grad
                for jj, loss_ in enumerate(loss):
                    last_ls[jj].append(loss_)
                    last_ls[jj] = last_ls[jj][-self.plateau_length:]
                    if last_ls[jj][-1] > last_ls[jj][0] and len(last_ls[jj]) == self.plateau_length:
                        if lr[jj] > self.min_lr:
                            lr[jj] = max(lr[jj] / self.plateau_drop, self.min_lr)
                        last_ls[jj] = []

                lr_t = torch.tensor(lr, device=adver_x.device, dtype=torch.float).unsqueeze(1).unsqueeze(2)
                adver_x.data = adver_x + self.grad_sign * lr_t * torch.sign(grad)
                adver_x.data = torch.min(torch.max(adver_x.data, lower), upper)

                iter_inner += 1
                n_iters += 1

            threshold += delta
            iter_outer += 1

    def estimate_threshold(self, x, step=0.1):
        if self.task == 'CSI':
            print("--- Warning: no need to estimate threshold for CSI, quitting ---")
            return

        with torch.no_grad():
            estimated_thresholds = []
            for xx in x.unsqueeze(0):  # parallel running, not easy for batch running
                estimated_threshold = self.estimate_threshold_run(xx, step)
                if estimated_threshold is not None:
                    estimated_thresholds.append(estimated_threshold)
            if len(estimated_thresholds) > 0:
                self.threshold = np.mean(estimated_thresholds)
            else:
                self.threshold = None
            return self.threshold


def test(model, inputs, labels, device):
    model.eval()
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
    accuracy = 100 * correct / total
    return accuracy


def calculate_snr(clean_wav, noisy_wav, epsilon=1e-10):
    noise_power = np.sum((clean_wav - noisy_wav) ** 2)
    clean_power = np.sum(clean_wav ** 2)
    snr = 10 * np.log10(clean_power / (noise_power + epsilon))
    return snr


def calculate_stoi(reference_waveform, degraded_waveform, sample_rate=16000):
    stoi_score = stoi(reference_waveform, degraded_waveform, sample_rate, extended=False)
    return stoi_score


def parse_args():
    fakebob_parser = argparse.ArgumentParser()
    fakebob_parser.add_argument('-targeted', action='store_true', default=False)
    fakebob_parser.add_argument('-target_label_file',
                                default=None)  # the path of the file containing the target label; generated by set_target_label.py
    fakebob_parser.add_argument('-batch_size', type=int, default=1)
    fakebob_parser.add_argument('-EOT_size', type=int, default=1)
    fakebob_parser.add_argument('-EOT_batch_size', type=int, default=1)
    fakebob_parser.add_argument('-start', type=int, default=0)
    fakebob_parser.add_argument('-end', type=int, default=-1)

    fakebob_parser.add_argument('--task', type=str, default='CSI',
                                choices=['CSI', 'SV', 'OSI'])  # the attack use this to set the loss function
    fakebob_parser.add_argument('--threshold', type=float, default=None)  # for SV/OSI task; real threshold of the model
    fakebob_parser.add_argument('--threshold_estimated', type=float,
                                default=None)  # for SV/OSI task; estimated threshold by FAKEBOB
    fakebob_parser.add_argument('--thresh_est_wav_path', type=str, nargs='+',
                                default=None)  # the audio path used to estimate the threshold, should from imposter (initially rejected)
    fakebob_parser.add_argument('--thresh_est_step', type=float,
                                default=0.1)  # the smaller, the accurate, but the slower
    fakebob_parser.add_argument('--confidence', type=float, default=0.)
    fakebob_parser.add_argument("--epsilon", "-epsilon", default=0.01, type=float)
    fakebob_parser.add_argument("--max_iter", "-max_iter", default=500, type=int)
    fakebob_parser.add_argument("--max_lr", "-max_lr", default=0.001, type=float)
    fakebob_parser.add_argument("--min_lr", "-min_lr", default=1e-6, type=float)
    fakebob_parser.add_argument("--samples_per_draw", "-samples", default=50, type=int)
    fakebob_parser.add_argument("--samples_batch", "-samples_batch", default=50, type=int)
    fakebob_parser.add_argument("--sigma", "-sigma", default=0.001, type=float)
    fakebob_parser.add_argument("--momentum", "-momentum", default=0.9, type=float)
    fakebob_parser.add_argument("--plateau_length", "-plateau_length", default=5, type=int)
    fakebob_parser.add_argument("--plateau_drop", "-plateau_drop", default=2.0, type=float)
    fakebob_parser.add_argument("--stop_early", "-stop_early", action='store_false', default=True)
    fakebob_parser.add_argument("--stop_early_iter", "-stop_early_iter", type=int, default=100)

    args = fakebob_parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESC_Net(num_classes=50)
    model.load_state_dict(torch.load('best_model_esc50.pt'))
    model.eval()
    model.to(device)
    model = defended_model(model)

    attacker = FAKEBOB(model, threshold=args.threshold_estimated, task=args.task, targeted=args.targeted,
                       confidence=args.confidence,
                       epsilon=args.epsilon, max_iter=args.max_iter, max_lr=args.max_lr,
                       min_lr=args.min_lr, samples_per_draw=args.samples_per_draw,
                       samples_per_draw_batch_size=args.samples_batch, sigma=args.sigma,
                       momentum=args.momentum, plateau_length=args.plateau_length,
                       plateau_drop=args.plateau_drop,
                       stop_early=args.stop_early, stop_early_iter=args.stop_early_iter,
                       batch_size=args.batch_size,
                       EOT_size=args.EOT_size, EOT_batch_size=args.EOT_batch_size,
                       verbose=1)

    total_accuracy = 0
    total_snr = 0
    total_stoi = 0
    num_batches = 0

    path_audio = "D:/ESC-50-master/audio"
    batch_size = 32
    test_frac = 0.2
    files = Path(path_audio).glob('[1-5]-*')
    items = [(str(file), file.name.split('-')[-1].replace('.wav', '')) for file in files]
    files = Path('ESC-50-augmented-data').glob('[1-5]-*')
    items += [(str(file), file.name.split('-')[-1].replace('.wav', '')) for file in files]

    length = len(items)
    indices = np.arange(length)
    np.random.shuffle(indices)

    split = int(np.floor(test_frac * length))
    train_indices, test_indices = indices[split:], indices[:split]

    test_data = DataGenerator(test_indices, items, kind='test')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for batch in test_loader:
        x, y = batch
        x=x.unsqueeze(1)
        x, y = x.to(device), y.to(device)
        adver, success = attacker.attack(x, y)
        accuracy = test(model, adver, y, device)
        total_accuracy += accuracy
        num_batches += 1

        print(success)
        adver_np = adver.cpu().numpy()
        origin_np = x.cpu().numpy()
        snr = calculate_snr(origin_np, adver_np)
        stoi_scores = []
        for j in range(origin_np.shape[0]):
            orig_sample = origin_np[j, 0, :]
            adv_sample = adver_np[j, 0, :]
            stoi_score = calculate_stoi(orig_sample, adv_sample)
            stoi_scores.append(stoi_score)

        avg_stoi_score = np.mean(stoi_scores)

        total_snr += snr
        total_stoi += avg_stoi_score

        print(f'SNR: {snr:.4f} dB')
        print(f'STOI: {avg_stoi_score:.4f}')
        print(f'Accuracy: {accuracy:.4f}')

    average_accuracy = total_accuracy / num_batches
    average_snr = total_snr / num_batches
    average_stoi = total_stoi / num_batches

    print(f'Average accuracy: {average_accuracy:.4f}%')
    print(f'Average SNR: {average_snr:.4f} dB')
    print(f'Average STOI: {average_stoi:.4f}')

    #
