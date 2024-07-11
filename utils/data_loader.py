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


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store class
            y[i] = self.labels[ID]

            # Store sample
            X[i, ] = np.load(f"{patch_dir}/{reverse_class_label[y[i]]}/" + ID + ".npy", mmap_mode='r+')

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes+1)


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
        self.vl_data = None
        self.vl_mask = None
        self.vl_labels = None
        self.vl_meta = None
        self.vl_cls = None

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
        # shuffle_buffer_size = 100
        if os.path.exists(os.path.join(npy_dir, f"test_tomo.npy")):
            self.ts_data = np.load(os.path.join(npy_dir, "test_tomo.npy"), mmap_mode='r+')
        if os.path.exists(os.path.join(npy_dir, f"test_mask.npy")):
            self.ts_mask = np.load(os.path.join(npy_dir, f"test_mask.npy"), mmap_mode='r+')
        self.ts_labels = np.load(os.path.join(npy_dir, f"test_{config_dict['representation_space']}.npy"), mmap_mode='r+')
        self.ts_meta = np.load(os.path.join(npy_dir, "test_meta.npy"), mmap_mode='r+')
        # print(self.ts_meta[0:5])
        # dataset = tf.data.Dataset.from_tensor_slices((self.ts_data, self.ts_labels, self.ts_mask, self.ts_meta))
        # # self.ts_data = dataset.shuffle(shuffle_buffer_size).batch(config_dict['batch_size'])
        # self.ts_mask.close()
        # print(self.ts_data)

        if config_dict['positions']:
            self.concat_pose()

        if config_dict['classification']:
            self.create_cls_labels()

    def concat_pose(self):
        train_position_labels = np.load(os.path.join(npy_dir, "train_shifts.npy"), mmap_mode='r+')
        test_position_labels = np.load(os.path.join(npy_dir, "test_shifts.npy"), mmap_mode='r+')

        self.labels = np.concatenate((self.labels, train_position_labels), axis=1)
        self.ts_labels = np.concatenate((self.labels, test_position_labels), axis=1)

    def create_cls_labels(self):

        self.cls = np.repeat(0, self.data.shape[0])
        self.ts_cls = np.repeat(0, self.ts_data.shape[0])

        tr_cls_data = np.load(os.path.join(cls_dir, "train_tomo.npy"), mmap_mode='r+')
        # if os.path.exists(os.path.join(cls_dir, f"train_mask.npy")):
        #     tr_cls_mask = np.load(os.path.join(cls_dir, f"train_mask.npy"), mmap_mode='r+')
        tr_cls_labels = np.load(os.path.join(cls_dir, f"train_{config_dict['representation_space']}.npy"), mmap_mode='r+')
        tr_cls_meta = np.load(os.path.join(cls_dir, "train_meta.npy"), mmap_mode='r+')

        if config_dict['num_samples'] != 0:
            tr_cls_data, tr_cls_labels = tr_cls_data[0:config_dict['num_samples'], :, :, :, :], \
                                         tr_cls_labels[0:config_dict['num_samples'], :]
            tr_cls_meta = tr_cls_meta[0:config_dict['num_samples'], :]
            # if os.path.exists(os.path.join(cls_dir, f"train_mask.npy")):
            #     tr_cls_mask = tr_cls_mask[0:config_dict['num_samples'], :, :, :, :]
        tr_cls = np.repeat(1, tr_cls_data.shape[0])

        ts_cls_data = np.load(os.path.join(cls_dir, "test_tomo.npy"), mmap_mode='r+')
        # if os.path.exists(os.path.join(cls_dir, f"test_mask.npy")):
        #     ts_cls_mask = np.load(os.path.join(cls_dir, f"test_mask.npy"), mmap_mode='r+')
        ts_cls_labels = np.load(os.path.join(cls_dir, f"test_{config_dict['representation_space']}.npy"), mmap_mode='r+')
        ts_cls_meta = np.load(os.path.join(cls_dir, "test_meta.npy"), mmap_mode='r+')
        ts_cls = np.repeat(1, ts_cls_data.shape[0])

        self.data = np.concatenate((self.data, tr_cls_data), axis=0)
        # self.mask = np.concatenate((self.mask, tr_cls_mask), axis=0)
        self.labels = np.concatenate((self.labels, tr_cls_labels), axis=0)
        self.meta = np.concatenate((self.meta, tr_cls_meta), axis=0)
        self.cls = np.concatenate((self.cls, tr_cls), axis=0)

        self.ts_data = np.concatenate((self.ts_data, ts_cls_data), axis=0)
        # self.ts_mask = np.concatenate((self.ts_mask, ts_cls_meta), axis=0)
        self.ts_labels = np.concatenate((self.ts_labels, ts_cls_labels), axis=0)
        self.ts_meta = np.concatenate((self.ts_meta, ts_cls_meta), axis=0)
        self.ts_cls = np.concatenate((self.ts_cls, ts_cls), axis=0)

        self.tr_steps = int(np.round(self.data.shape[0] * (1 - config_dict['valid_prc'])) /
                            config_dict['batch_size_train'])
        self.vl_steps = int(np.round(self.data.shape[0] * config_dict['valid_prc']) /
                            config_dict['batch_size_valid'])

    def concat_data(self, data_path, label_path, data_mode="train"):
        next_data = np.load(data_path)
        next_labels = np.load(label_path)

        if data_mode == "train":
            self.data = np.concatenate((self.data, next_data), axis=0)
            self.labels = np.concatenate((self.labels, next_labels), axis=0)
        elif data_mode == "test":
            self.ts_data = np.concatenate((self.ts_data, next_data), axis=0)
            self.ts_labels = np.concatenate((self.ts_labels, next_labels), axis=0)

    def load_batch(self, b, mode="train"):
        cnt = 0
        bstart = b * config_dict['batch_size_train']
        bend = bstart + config_dict['batch_size_train']

        batch_data = np.zeros((config_dict['batch_size_train'],
                               config_dict['patch_size'],
                               config_dict['patch_size'],
                               config_dict['patch_size'], 1), dtype=np.float32)
        # batch_mask = np.zeros((config_dict['batch_size_train'],
        #                        config_dict['patch_size'],
        #                        config_dict['patch_size'],
        #                        config_dict['patch_size'], self.tr_mask.shape[-1]), dtype=np.int8)
        batch_label = np.zeros((config_dict['batch_size_train'], config_dict['output_dim']), dtype=np.float32)
        # batch_meta = np.zeros((config_dict['batch_size_train'], 4), dtype=np.float32)

        if b == 0:
            assert len(self.data) == len(self.labels)
            p = np.random.permutation(len(self.data))
            self.tr_data, self.tr_labels = self.data[p], self.labels[p]
            self.tr_meta = self.meta[p]
            # self.tr_mask = self.mask[p]

            num_train_samples = int(np.round(len(self.tr_data) * (1 - config_dict['valid_prc'])))

            self.vl_data, self.vl_labels = self.tr_data[num_train_samples:], self.tr_labels[num_train_samples:]
            self.vl_meta = self.tr_meta[num_train_samples:]

            self.tr_data, self.tr_labels = self.tr_data[0:num_train_samples], self.tr_labels[0:num_train_samples]
            self.tr_meta = self.tr_meta[0:num_train_samples]

        data, labels, metas = self.tr_data, self.tr_labels, self.tr_meta
        if mode == "validation":
            data, labels, metas = self.vl_data, self.vl_labels, self.vl_meta

        print(f"----- batch {bstart} - {bend} -----")
        for i in range(bstart, bend):

            # add the sample and its label to the current training batch
            batch_data[cnt, :, :, :, :] = data[i, :, :, :]
            # batch_mask[cnt, :, :, :, :] = mask[i, :, :, :]
            batch_label[cnt] = labels[i]
            # batch_meta[cnt] = meta[i]
            cnt = cnt + 1
        return batch_data, batch_label

    def split_data(self):
        x_train, x_valid, y_train, y_valid = train_test_split(self.data,
                                                              self.labels,
                                                              test_size=config_dict['valid_prc'],
                                                              shuffle=True)
        return x_train, x_valid, y_train, y_valid

    def hybrid_split_data(self):
        if not config_dict['classification']:
            x_train, x_valid, y_train_reg, y_valid_reg, y_train_cls, y_valid_cls = train_test_split(self.data,
                                                                                                    self.labels,
                                                                                                    self.mask,
                                                                                                    test_size=config_dict['valid_prc'],
                                                                                                    shuffle=True)
        else:
            x_train, x_valid, y_train_reg, y_valid_reg, y_train_cls, y_valid_cls = train_test_split(self.data,
                                                                                                    self.labels,
                                                                                                    self.cls,
                                                                                                    test_size=config_dict['valid_prc'],
                                                                                                    shuffle=True)

        return x_train, x_valid, y_train_reg, y_valid_reg, y_train_cls, y_valid_cls

    def fold_data(self, k):
        folds = list(StratifiedKFold(n_splits=k,
                                     shuffle=True,
                                     random_state=seed_num).split(self.data,
                                                                  self.labels,))
        return folds

    def efficient_data_loader(self, b, bsize=config_dict['batch_size_train'], mode="train"):
        cnt = 0
        bstart = b * bsize
        bend = bstart + bsize

        batch_data = np.zeros((bsize,
                               config_dict['patch_size'],
                               config_dict['patch_size'],
                               config_dict['patch_size'], 1), dtype=np.float32)
        batch_mask = np.zeros((config_dict['batch_size_train'],
                               config_dict['patch_size'],
                               config_dict['patch_size'],
                               config_dict['patch_size'], config_dict['num_classes']), dtype=np.int8)
        batch_label = np.zeros((bsize, config_dict['output_dim']), dtype=np.float32)
        batch_meta = []

        if b == 0:
            if config_dict['num_samples'] != 0:
                self.labels = self.labels[0:config_dict['num_samples'], :]
                self.meta = self.meta[0:config_dict['num_samples'], :]
            p = np.random.permutation(len(self.meta))
            self.tr_labels = self.labels[p]
            self.tr_meta = self.meta[p]

            num_train_samples = int(np.round(len(self.meta) * (1 - config_dict['valid_prc'])))
            self.vl_labels = self.tr_labels[num_train_samples:]
            self.vl_meta = self.tr_meta[num_train_samples:]

            self.tr_labels = self.tr_labels[0:num_train_samples]
            self.tr_meta = self.tr_meta[0:num_train_samples]

        labels, metas, fpath = self.tr_labels, self.tr_meta, mode
        if mode == "validation":
            labels, metas = self.vl_labels, self.vl_meta
            fpath = "train"
        if mode == "test":
            labels, metas = self.ts_labels, self.ts_meta

        for i in range(bstart, bend):
            # add the sample and its label to the current training batch
            rtomo = np.load(os.path.join(patch_dir, f"{fpath}/{metas[i, 5]}"), mmap_mode='r+', allow_pickle=True)
            rmask = np.load(os.path.join(patch_dir, f"{fpath}/{metas[i, 6]}"), mmap_mode='r+', allow_pickle=True)

            batch_data[cnt, :, :, :, :] = np.expand_dims(rtomo, axis=len(rtomo.shape))
            batch_mask[cnt, :, :, :, :] = np.expand_dims(rmask, axis=len(rmask.shape))
            batch_label[cnt] = labels[i]
            batch_meta.append(metas[i])
            cnt = cnt + 1

        return batch_data, batch_label



