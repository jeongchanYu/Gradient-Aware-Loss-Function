import os, sys, random, math
import numpy as np
import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import util_func as util


# speech enhancement dataset
class train_dataset(Dataset):
    def __init__(self, dataset_path, frame_size, sample_rate):
        dataset_path = dataset_path.replace('~', os.path.expanduser('~'))
        dataset_path = os.path.join(dataset_path, 'train')
        self.clean_path = os.path.normpath(os.path.abspath(os.path.join(dataset_path, 'clean')))
        self.noisy_path = os.path.normpath(os.path.abspath(os.path.join(dataset_path, 'noisy')))
        self.clean_path_list = util.load_file_path_list(self.clean_path, 'wav')
        self.noisy_path_list = util.load_file_path_list(self.noisy_path, 'wav')

        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.number_of_files = len(self.clean_path_list)

    def __len__(self):
        return self.number_of_files

    def __getitem__(self, index):
        clean_ = librosa.load(self.clean_path_list[index], sr=self.sample_rate, dtype='float32')[0]
        noisy_ = librosa.load(self.noisy_path_list[index], sr=self.sample_rate, dtype='float32')[0]

        clean = np.zeros(len(clean_)+self.frame_size*2, dtype='float32')
        noisy = np.zeros(len(noisy_)+self.frame_size*2, dtype='float32')
        clean[self.frame_size:self.frame_size+len(clean_)] += clean_
        noisy[self.frame_size:self.frame_size+len(noisy_)] += noisy_

        rand_index = np.random.randint(0, len(clean)-self.frame_size+1)
        clean = clean[rand_index:rand_index+self.frame_size]
        noisy = noisy[rand_index:rand_index+self.frame_size]
        return [self.noisy_path_list[index].replace(self.noisy_path, '').lstrip('./\\'),
                torch.from_numpy(clean),
                torch.from_numpy(noisy)]

class test_dataset(Dataset):
    def __init__(self, dataset_path, sample_rate):
        dataset_path = dataset_path.replace('~', os.path.expanduser('~'))
        dataset_path = os.path.join(dataset_path, 'test')
        self.clean_path = os.path.normpath(os.path.abspath(os.path.join(dataset_path, 'clean')))
        self.noisy_path = os.path.normpath(os.path.abspath(os.path.join(dataset_path, 'noisy')))
        self.clean_path_list = util.load_file_path_list(self.clean_path, 'wav')
        self.noisy_path_list = util.load_file_path_list(self.noisy_path, 'wav')

        self.number_of_files = len(self.clean_path_list)
        self.sample_rate = sample_rate

    def __len__(self):
        return self.number_of_files

    def __getitem__(self, index):
        clean = librosa.load(self.clean_path_list[index], sr=self.sample_rate, dtype=np.float32)[0]
        noisy = librosa.load(self.noisy_path_list[index], sr=self.sample_rate, dtype=np.float32)[0]
        return [self.noisy_path_list[index].replace(self.noisy_path, '').lstrip('./\\'),
                torch.from_numpy(clean),
                torch.from_numpy(noisy)]