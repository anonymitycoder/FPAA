import os
import random

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class Esc50Dataset(Dataset):
    def __init__(self, data_path, train=True, transform=None, test_size=0.2, random_state=1):
        metadata_path = os.path.join(data_path, "meta", "esc50.csv")
        self.data = pd.read_csv(metadata_path)
        self.file_paths = []
        self.labels = []
        for i in range(len(self.data)):
            file_path = os.path.join(data_path, "audio", self.data.iloc[i]["filename"])
            label = self.data.iloc[i]["target"]
            self.file_paths.append(file_path)
            self.labels.append(label)
        self.transform = transform
        self.train = train
        train_indices, test_indices = train_test_split(
            list(range(len(self.data))),
            test_size=test_size,
            random_state=random_state
        )
        if self.train:
            self.file_paths = [self.file_paths[i] for i in train_indices]
            self.labels = [self.labels[i] for i in train_indices]
        else:
            self.file_paths = [self.file_paths[i] for i in test_indices]
            self.labels = [self.labels[i] for i in test_indices]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(file_path, sr=22050)
        stft = librosa.stft(y=audio, n_fft=2048, hop_length=512)
        stft_magnitude, stft_phase = librosa.magphase(stft)
        stft_db = np.expand_dims(stft_magnitude, axis=0)
        stft_db = torch.from_numpy(stft_db).float()
        if self.transform:
            stft_db = self.transform(stft_db)
        return stft_db, label


class ESC50OneClass(Dataset):
    def __init__(self, data_path, class_num):
        metadata_path = os.path.join(data_path, "meta", "esc50.csv")
        self.data = pd.read_csv(metadata_path)
        self.data = self.data[self.data["target"] == class_num].reset_index(drop=True)
        self.file_paths = []
        self.labels = []
        for i in range(len(self.data)):
            file_path = os.path.join(data_path, "audio", self.data.iloc[i]["filename"])
            label = self.data.iloc[i]["target"]
            self.file_paths.append(file_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(file_path, sr=22050)
        stft = librosa.stft(y=audio, n_fft=2048, hop_length=512)
        stft_magnitude, stft_phase = librosa.magphase(stft)
        stft_phase = np.angle(stft_phase)
        stft_db_magnitude = np.expand_dims(stft_magnitude, axis=0)
        stft_db_phase = np.expand_dims(stft_phase, axis=0)
        stft_db_magnitude = torch.from_numpy(stft_db_magnitude).float()
        stft_db_phase = torch.from_numpy(stft_db_phase).float()
        return stft_db_magnitude, stft_db_phase, label


class UrbanSound8KDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None, test_size=0.2, random_state=42):
        metadata_path = os.path.join(data_path, "metadata", "UrbanSound8K.csv")
        self.data = pd.read_csv(metadata_path)
        self.file_paths = []
        self.labels = []
        for i in range(len(self.data)):
            file_path = os.path.join(data_path, "audio", "fold" + str(self.data.iloc[i]["fold"]) + "/",
                                     self.data.iloc[i]["slice_file_name"])
            label = self.data.iloc[i]["classID"]
            self.file_paths.append(file_path)
            self.labels.append(label)
        self.transform = transform
        self.train = train
        train_indices, test_indices = train_test_split(
            list(range(len(self.data))),
            test_size=test_size,
            random_state=random_state
        )
        self.file_paths = [self.file_paths[i] for i in train_indices] if self.train else [self.file_paths[i] for i in
                                                                                          test_indices]
        self.labels = [self.labels[i] for i in train_indices] if self.train else [self.labels[i] for i in test_indices]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(file_path, sr=22050)
        if audio.shape[0] > 88200:
            max_audio_start = audio.shape[0] - 88200
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start: audio_start + 88200]
        else:
            audio = librosa.util.pad_center(audio, size=88200)
        stft = librosa.stft(y=audio, n_fft=2048, hop_length=512)
        stft_magnitude, stft_phase = librosa.magphase(stft)
        stft_db = np.expand_dims(stft_magnitude, axis=0)
        stft_db = torch.from_numpy(stft_db).float()
        if self.transform:
            stft_db = self.transform(stft_db)
        return stft_db, label


class UrbanSound8KOneClass(Dataset):
    def __init__(self, data_path, class_num):
        metadata_path = os.path.join(data_path, "metadata", "UrbanSound8K.csv")
        self.data = pd.read_csv(metadata_path)
        self.data = self.data.groupby("classID").apply(lambda x: x[x["classID"] == class_num]).reset_index(
            drop=True)
        self.file_paths = []
        self.labels = []
        for i in range(len(self.data)):
            file_path = os.path.join(data_path, "audio", "fold" + str(self.data.iloc[i]["fold"]) + "/",
                                     self.data.iloc[i]["slice_file_name"])
            label = self.data.iloc[i]["classID"]
            self.file_paths.append(file_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(file_path, sr=22050)
        if audio.shape[0] > 88200:
            max_audio_start = audio.shape[0] - 88200
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start: audio_start + 88200]
        else:
            audio = librosa.util.pad_center(audio, size=88200)
        stft = librosa.stft(y=audio, n_fft=2048, hop_length=512)
        stft_magnitude, stft_phase = librosa.magphase(stft)
        stft_phase = np.angle(stft_phase)
        stft_db_magnitude = np.expand_dims(stft_magnitude, axis=0)
        stft_db_phase = np.expand_dims(stft_phase, axis=0)
        stft_db_magnitude = torch.from_numpy(stft_db_magnitude).float()
        stft_db_phase = torch.from_numpy(stft_db_phase).float()
        return stft_db_magnitude, stft_db_phase, label
