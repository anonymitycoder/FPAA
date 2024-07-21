import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from abc import ABCMeta, abstractmethod

from pystoi import stoi
from torch import nn
from torch.utils.data import DataLoader

from defended_model import defended_model
from model import CNN, ESC_Net
from train_ESC50 import DataGenerator


class SEC4SR_CrossEntropy(nn.CrossEntropyLoss):

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


class SEC4SR_MarginLoss(nn.Module):

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


def resolve_prediction(decisions):
    predict = []
    for d in decisions:
        counts = Counter(d)
        predict.append(counts.most_common(1)[0][0])
    predict = np.array(predict)
    return predict


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


class Attack(metaclass=ABCMeta):

    @abstractmethod
    def attack(self, x, y, verbose=1, EOT_size=1, EOT_batch_size=1):
        pass

    def compare(self, y, y_pred, targeted):
        if targeted:
            return (y_pred == y).tolist()
        else:
            return (y_pred != y).tolist()


class SirenAttack(Attack):

    def __init__(self, model, threshold=None,
                 task='CSI', targeted=False, confidence=0.,
                 epsilon=0.002, max_epoch=300, max_iter=30,
                 c1=1.4961, c2=1.4961, n_particles=25, w_init=0.9, w_end=0.1,
                 batch_size=1, EOT_size=1, EOT_batch_size=1, verbose=1, abort_early=True, abort_early_iter=10,
                 abort_early_epoch=10):

        self.model = model
        self.threshold = threshold
        self.task = task
        self.targeted = targeted
        self.confidence = confidence
        self.epsilon = epsilon
        self.max_epoch = max_epoch
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.n_particles = n_particles
        self.w_init = w_init
        self.w_end = w_end
        self.batch_size = batch_size
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.verbose = verbose

        self.abort_early = abort_early
        self.abort_early_iter = abort_early_iter
        self.abort_early_epoch = abort_early_epoch

    def attack_batch(self, x_batch, y_batch, lower, upper, batch_id):

        with torch.no_grad():

            v_upper = torch.abs(lower - upper)
            v_lower = -v_upper

            x_batch_clone = x_batch.clone()  # for return
            n_audios, n_channels, N = x_batch.shape
            consider_index = list(range(n_audios))
            # pbest_locations = np.random.uniform(low=lower.unsqueeze(1).cpu().numpy(),
            #                     high=upper.unsqueeze(1).cpu().numpy(), size=(n_audios, self.n_particles, n_channels, N))
            # pbest_locations = torch.tensor(pbest_locations, device=x_batch.device, dtype=torch.float)

            gbest_location = torch.zeros(n_audios, n_channels, N, dtype=torch.float, device=x_batch.device)
            gbests = torch.ones(n_audios, device=x_batch.device, dtype=torch.float) * np.infty
            gbest_predict = np.array([None] * n_audios)
            prev_gbest = gbests.clone()
            prev_gbest_epoch = gbests.clone()

            continue_flag = True
            for epoch in range(self.max_epoch):

                if not continue_flag:
                    break

                if epoch == 0:
                    pbest_locations = np.random.uniform(low=lower.unsqueeze(1).cpu().numpy(),
                                                        high=upper.unsqueeze(1).cpu().numpy(),
                                                        size=(n_audios, self.n_particles, n_channels, N))
                    pbest_locations = torch.tensor(pbest_locations, device=x_batch.device, dtype=torch.float)
                    pbests = torch.ones(n_audios, self.n_particles, device=x_batch.device, dtype=torch.float) * np.infty
                else:
                    best_index = torch.argmin(pbests, dim=1)  # (len(consider_index), )
                    best_location = pbest_locations[
                        np.arange(len(consider_index)), best_index]  # (len(consider_index), n_channels, N)
                    pbest_locations = np.random.uniform(low=lower.unsqueeze(1).cpu().numpy(),
                                                        high=upper.unsqueeze(1).cpu().numpy(),
                                                        size=(len(consider_index), self.n_particles - 1, n_channels, N))
                    pbest_locations = torch.tensor(pbest_locations, device=x_batch.device, dtype=torch.float)
                    pbest_locations = torch.cat((best_location.unsqueeze(1), pbest_locations), dim=1)
                    pbests_new = torch.ones(len(consider_index), self.n_particles - 1, device=x_batch.device,
                                            dtype=torch.float) * np.infty
                    pbests = torch.cat((pbests[np.arange(len(consider_index)), best_index].unsqueeze(1), pbests_new),
                                       dim=1)

                locations = pbest_locations.clone()
                # volicities = np.random.uniform(low=lower.unsqueeze(1).cpu().numpy(),
                #                 high=upper.unsqueeze(1).cpu().numpy(), size=(len(consider_index), self.n_particles, n_channels, N))
                volicities = np.random.uniform(low=v_lower.unsqueeze(1).cpu().numpy(),
                                               high=v_upper.unsqueeze(1).cpu().numpy(),
                                               size=(len(consider_index), self.n_particles, n_channels, N))
                volicities = torch.tensor(volicities, device=x_batch.device, dtype=torch.float)

                ### ????
                # pbests = torch.ones(len(consider_index), self.n_particles, device=x_batch.device, dtype=torch.float) * np.infty

                continue_flag_inner = True

                # for iter in range(self.max_iter):
                for iter in range(self.max_iter + 1):

                    if not continue_flag_inner:
                        break

                    eval_x = locations + x_batch.unsqueeze(1)  # (n_audios, self.n_particles, n_channels, N)
                    eval_x = eval_x.view(-1, n_channels, N)
                    eval_y = None
                    for jj, y_ in enumerate(y_batch):
                        tmp = torch.tensor([y_] * self.n_particles, dtype=torch.long, device=x_batch.device)
                        if jj == 0:
                            eval_y = tmp
                        else:
                            eval_y = torch.cat((eval_y, tmp))
                    # print(eval_x.shape, eval_y.shape)
                    _, loss, _, decisions = self.EOT_wrapper(eval_x, eval_y)
                    EOT_num_batches = int(self.EOT_wrapper.EOT_size // self.EOT_wrapper.EOT_batch_size)
                    loss.data /= EOT_num_batches  # (n_audios*n_p,)
                    loss = loss.view(len(consider_index), -1)  # (n_audios, n_p)
                    predict = resolve_prediction(decisions).reshape(len(consider_index), -1)

                    update_index = torch.where(loss < pbests)
                    update_ii = update_index[0].cpu().numpy().tolist()
                    update_jj = update_index[1].cpu().numpy().tolist()
                    if len(update_ii) > 0:
                        for ii, jj in zip(update_ii, update_jj):
                            pbests[ii, jj] = loss[ii, jj]
                            pbest_locations[ii, jj, ...] = locations[ii, jj, ...]

                    # if self.abort_early and (iter+1) % self.abort_early_iter == 0:
                    #     prev_gbest.data = gbests

                    gbest_index = torch.argmin(pbests, 1)
                    for kk in range(gbest_index.shape[0]):
                        index = consider_index[kk]
                        if pbests[kk, gbest_index[kk]] < gbests[index]:
                            gbests[index] = pbests[kk, gbest_index[kk]]
                            gbest_location[index] = pbest_locations[kk, gbest_index[kk]]
                            gbest_predict[index] = predict[kk, gbest_index[kk]]

                    if self.verbose:
                        print('batch: {}, epoch: {}, iter: {}, y: {}, y_pred: {}, gbest: {}'.format(batch_id,
                                                                                                    epoch, iter,
                                                                                                    y_batch.cpu().numpy().tolist(),
                                                                                                    gbest_predict[
                                                                                                        consider_index],
                                                                                                    gbests[
                                                                                                        consider_index].cpu().numpy().tolist()))

                    if self.abort_early and (iter + 1) % self.abort_early_iter == 0:
                        if torch.mean(gbests) > 0.9999 * torch.mean(prev_gbest):
                            print('Converge, Break Inner Loop')
                            continue_flag_inner = False
                            # break
                        # prev_gbest.data = gbests
                        prev_gbest = gbests.clone()

                    # stop early
                    # x_batch, y_batch, lower, upper
                    # pbest_locations, locations, v, pbests
                    # consider_index
                    # delete alrady found examples
                    x_batch, y_batch, lower, upper, \
                        pbest_locations, locations, volicities, pbests, \
                        consider_index = self.delete_found(gbests[consider_index], x_batch, y_batch, lower, upper,
                                                           pbest_locations, locations, volicities, pbests,
                                                           consider_index)
                    if len(consider_index) == 0:
                        continue_flag = False  # used to break the outer loop
                        break
                    else:
                        v_upper = torch.abs(lower - upper)
                        v_lower = -v_upper

                    if iter < self.max_iter:
                        w = (self.w_init - self.w_end) * (self.max_iter - iter - 1) / self.max_iter + self.w_end
                        # r1 = np.random.rand() + 0.00001
                        # r2 = np.random.rand() + 0.00001
                        r1 = np.random.rand(len(consider_index), self.n_particles, n_channels, N) + 0.00001
                        r2 = np.random.rand(len(consider_index), self.n_particles, n_channels, N) + 0.00001
                        r1 = torch.tensor(r1, device=x_batch.device, dtype=torch.float)
                        r2 = torch.tensor(r2, device=x_batch.device, dtype=torch.float)
                        volicities = (w * volicities + self.c1 * r1 * (pbest_locations - locations) +
                                      self.c2 * r2 * (gbest_location[consider_index, ...].unsqueeze(1) - locations))
                        locations = locations + volicities
                        locations = torch.min(torch.max(locations, lower.unsqueeze(1)), upper.unsqueeze(1))

                if self.abort_early and (epoch + 1) % self.abort_early_epoch == 0:
                    if torch.mean(gbests) > 0.9999 * torch.mean(prev_gbest_epoch):
                        print('Converge, Break Outer Loop')
                        continue_flag = False
                        # break
                    prev_gbest_epoch = gbests.clone()

            success = [False] * n_audios
            for kk, best_l in enumerate(gbests):
                if best_l < 0:
                    success[kk] = True

            return gbest_location + x_batch_clone, success

    def delete_found(self, gbests, x_batch, y_batch, lower, upper,
                     pbest_locations, locations, volicities, pbests,
                     consider_index):

        x_batch_u = None
        y_batch_u = None
        lower_u = None
        upper_u = None
        pbest_locations_u = None
        locations_u = None
        volicities_u = None
        pbests_u = None
        consider_index_u = []

        for ii, g in enumerate(gbests):
            if g < 0:
                continue
            else:
                if x_batch_u is None:
                    x_batch_u = x_batch[ii:ii + 1]
                    y_batch_u = y_batch[ii:ii + 1]
                    lower_u = lower[ii:ii + 1]
                    upper_u = upper[ii:ii + 1]
                    pbest_locations_u = pbest_locations[ii:ii + 1]
                    locations_u = locations[ii:ii + 1]
                    volicities_u = volicities[ii:ii + 1]
                    pbests_u = pbests[ii:ii + 1]
                else:
                    x_batch_u = torch.cat((x_batch_u, x_batch[ii:ii + 1]), 0)
                    y_batch_u = torch.cat((y_batch_u, y_batch[ii:ii + 1]))
                    lower_u = torch.cat((lower_u, lower[ii:ii + 1]), 0)
                    upper_u = torch.cat((upper_u, upper[ii:ii + 1]), 0)
                    pbest_locations_u = torch.cat((pbest_locations_u, pbest_locations[ii:ii + 1]), 0)
                    locations_u = torch.cat((locations_u, locations[ii:ii + 1]), 0)
                    volicities_u = torch.cat((volicities_u, volicities[ii:ii + 1]), 0)
                    pbests_u = torch.cat((pbests_u, pbests[ii:ii + 1]), 0)
                index = consider_index[ii]
                consider_index_u.append(index)

        return x_batch_u, y_batch_u, lower_u, upper_u, \
            pbest_locations_u, locations_u, volicities_u, pbests_u, \
            consider_index_u

    def attack(self, x, y):

        if self.task in ['SV', 'OSI'] and self.threshold is None:
            raise NotImplementedError('You are running black box attack for {} task, \
                        but the threshold not specified. Consider Estimating the threshold by FAKEBOB!')
        self.loss, self.grad_sign = resolve_loss('Margin', self.targeted, self.confidence, self.task, self.threshold,
                                                 False)
        self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, False)

        lower = -2
        upper = 2
        assert lower <= x.max() < upper, 'generating adversarial examples should be done in [-1, 1) float domain'
        n_audios, n_channels, _ = x.size()
        assert n_channels == 1, 'Only Support Mono Audio'
        assert y.shape[0] == n_audios, 'The number of x and y should be equal'
        # upper = torch.clamp(x+self.epsilon, max=upper)
        # lower = torch.clamp(x-self.epsilon, min=lower)
        lower = torch.clamp(-1 - x, min=-self.epsilon)  # for distortion, not adver audio
        upper = torch.clamp(1 - x, max=self.epsilon)  # for distortion, not adver audio

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--targeted', action='store_true', default=False)
    parser.add_argument('--target_label_file',
                        default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--EOT_size', type=int, default=1)
    parser.add_argument('--EOT_batch_size', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

    parser.add_argument('--task', type=str, default='CSI',
                        choices=['CSI', 'SV', 'OSI'])
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--threshold_estimated', type=float,
                        default=None)
    parser.add_argument('--thresh_est_wav_path', type=str, nargs='+',
                        default=None)
    parser.add_argument('--thresh_est_step', type=float,
                        default=0.1)
    parser.add_argument('--confidence', type=float, default=0.)
    parser.add_argument("--epsilon", "-epsilon", default=0.01, type=float)
    parser.add_argument("--max_lr", "-max_lr", default=0.01, type=float)
    parser.add_argument("--min_lr", "-min_lr", default=1e-6, type=float)
    parser.add_argument("--samples_per_draw", "-samples", default=50, type=int)
    parser.add_argument("--samples_batch", "-samples_batch", default=50, type=int)
    parser.add_argument("--sigma", "-sigma", default=0.001, type=float)
    parser.add_argument("--momentum", "-momentum", default=0.9, type=float)
    parser.add_argument("--plateau_length", "-plateau_length", default=5, type=int)
    parser.add_argument("--plateau_drop", "-plateau_drop", default=2.0, type=float)
    parser.add_argument("--stop_early", "-stop_early", action='store_false', default=True)
    parser.add_argument("--stop_early_iter", "-stop_early_iter", type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-EOT_size', type=int, default=1)
    parser.add_argument('-EOT_batch_size', type=int, default=1)

    parser.add_argument("-max_epoch", default=10, type=int)
    parser.add_argument("-max_iter", default=30, type=int)
    parser.add_argument("-c1", type=float, default=1.4961)
    parser.add_argument("-c2", type=float, default=1.4961)
    parser.add_argument("-n_particles", default=50, type=int)
    parser.add_argument("-w_init", type=float, default=0.9)
    parser.add_argument("-w_end", type=float, default=0.1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESC_Net(num_classes=50)
    model.load_state_dict(torch.load('best_model_esc50.pt'))
    model.eval()
    model.to(device)
    model = defended_model(model)

    attacker = SirenAttack(model, threshold=args.threshold_estimated,
                           task=args.task, targeted=args.targeted, confidence=args.confidence,
                           epsilon=args.epsilon, max_epoch=args.max_epoch, max_iter=args.max_iter,
                           c1=args.c1, c2=args.c2, n_particles=args.n_particles, w_init=args.w_init, w_end=args.w_end,
                           batch_size=args.batch_size, EOT_size=args.EOT_size, EOT_batch_size=args.EOT_batch_size, )

    total_accuracy = 0
    total_snr = 0
    total_stoi = 0
    num_batches = 0

    path_audio = "D:/ESC-50-master/audio"
    batch_size = 2
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
        x = x.unsqueeze(1)
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
