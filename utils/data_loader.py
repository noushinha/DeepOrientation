import numpy as np
import os.path
from config import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils.utility_functions import write_mrc
import tensorflow as tf
import time
from datetime import timedelta
import keras
import pandas as pd

class DataLoader:
    def __init__(self):
        self.data = None
        self.mask = None
        self.labels = None
        self.meta = None
        self.cls = None

        self.tr_data = None
        self.tr_mask = None
        self.tr_labels = None
        self.tr_meta = None
        self.tr_cls = None
        self.ts_data = None
        self.ts_mask = None
        self.ts_labels = None
        self.ts_meta = None
        self.ts_cls = None

        self.tr_steps = 0
        self.vl_steps = 0

        self.load_train()
        self.load_test()

    def load_train(self):
        start_time = time.time()
        if os.path.exists(os.path.join(npy_dir, f"train_tomo.npy")):
            self.data = np.load(os.path.join(npy_dir, "train_tomo.npy"), mmap_mode='r+')
        if os.path.exists(os.path.join(npy_dir, f"train_mask.npy")):
            self.mask = np.load(os.path.join(npy_dir, f"train_mask.npy"), mmap_mode='r+')
        self.labels = np.load(os.path.join(npy_dir, f"train_{config_dict['representation_space']}.npy"), mmap_mode='r+')
        self.meta = np.load(os.path.join(npy_dir, "train_meta.npy"), mmap_mode='r+')

        if config_dict['num_samples'] != 0:
            self.labels = self.labels[0:config_dict['num_samples'], :]
            self.meta = self.meta[0:config_dict['num_samples'], :]
            if os.path.exists(os.path.join(npy_dir, f"train_tomo.npy")):
                self.data = self.data[0:config_dict['num_samples'], :, :, :, :]
            if os.path.exists(os.path.join(npy_dir, f"train_mask.npy")):
                self.mask = self.mask[0:config_dict['num_samples'], :, :, :, :]
        self.tr_steps = int(np.round(len(self.meta) * (1 - config_dict['valid_prc'])) /
                            config_dict['batch_size_train'])
        self.vl_steps = int(np.round(len(self.meta) * config_dict['valid_prc']) /
                            config_dict['batch_size_valid'])

        end_time = time.time()
        total_time = end_time - start_time
        deltatime = timedelta(seconds=total_time)
        print(f"Fetched train data in : {str(deltatime)}")

    def load_test(self):
        if os.path.exists(os.path.join(npy_dir, f"test_tomo.npy")):
            self.ts_data = np.load(os.path.join(npy_dir, "test_tomo.npy"), mmap_mode='r+')
        if os.path.exists(os.path.join(npy_dir, f"test_mask.npy")):
            self.ts_mask = np.load(os.path.join(npy_dir, f"test_mask.npy"), mmap_mode='r+')
        self.ts_labels = np.load(os.path.join(npy_dir, f"test_{config_dict['representation_space']}.npy"), mmap_mode='r+')
        self.ts_meta = np.load(os.path.join(npy_dir, "test_meta.npy"), mmap_mode='r+')

        if config_dict['positions']:
            self.concat_pose()

        if config_dict['classification']:
            self.create_cls_labels()

    def split_data(self):
        x_train, x_valid, y_train, y_valid = train_test_split(self.data,
                                                              self.labels,
                                                              test_size=config_dict['valid_prc'],
                                                              shuffle=True)
        return x_train, x_valid, y_train, y_valid




