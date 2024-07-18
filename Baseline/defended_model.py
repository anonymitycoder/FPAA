import torch
import torch.nn.functional as F
import torch.nn as nn

# from utils import *

# ensemble manner of multiple defenses
sequential = 'sequential'  # model(d_n(...d_2(d_(x))))
average = 'average'  # average(model(d_(x)), ..., model(d_n(x)))


class defended_model(nn.Module):

    def __init__(self, base_model, defense=None):
        super().__init__()

        self.base_model = base_model
        self.threshold = base_model.threshold

        self.defense = defense

    def process_sequential(self, x):
        return x

    def embedding(self, x):
        '''
        x: wav with shape [B, 1, T]
        return the same thing as the base model
        '''
        return x

    def forward(self, x):
        '''
        x: wav with shape [B, 1, T]
        return the same thing as the base model
        '''
        return self.base_model(x)

    def score(self, x):
        '''
        x: wav with shape [B, 1, T]
        return the same thing as the base model
        '''
        return F.softmax(self.base_model(x))

    def make_decision(self, x, enroll_embs=None):
        '''
        x: wav with shape [B, 1, T]
        return the same thing as the base model
        '''
        scores = self.score(x)

        decisions = torch.argmax(scores, dim=1)
        max_scores = torch.max(scores, dim=1)[0]
        decisions = torch.where(max_scores > self.base_model.threshold, decisions,
                                torch.tensor([-1] * decisions.shape[0], dtype=torch.int64, device=decisions.device))

        return decisions, scores
