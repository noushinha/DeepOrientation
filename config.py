import os
import numpy as np
from pathlib import Path


ROOT_DIR = Path(__file__).parent
seed_num = 42
np.random.seed(seed_num)
config_dict = {'num_layers': 4,
               'num_nodes': 128,
               'patch_size': 40,
               'shift_size': 2,
               'batch_size_train': 32,
               'batch_size_valid': 32,
               'valid_prc': .15,
               'test_prc': .1,
               'num_classes': 3,  # BG, PT, RB
               'initial_lr': 0.0001,
               'num_epochs': 15,
               'output_dim': 6,  # number of target variables in regressor head
               'num_samples': 0,
               'representation_space': "6dof",  # 6dof, quat, euler, shifts
               'lossname': "s.huber",  # point_wise_l2, huber
               'optname': "ADAM",  # ADAM, SGD
               'metricslist': ["mse", "mae"],
               'list_layers': ["fc1", "fc2", "fc3", "fc4"],
               'classification': False,
               'positions': False,
               'inference': False,
               'transfer_learning': False,
               'save_features': False,
               'save_filters': False,
               'checkpoint_dir': '',
               'data_type': 'polnet'}

config_dict['denselayer'] = np.repeat(config_dict['num_nodes'], config_dict['num_layers'])
base_dir = r'<path to root directory of DeepOrientation>'
npy_dir = os.path.join(base_dir, f"data/{config_dict['data_type']}/npy")
output_dir = os.path.join(base_dir, f"data/{config_dict['data_type']}/res")
log_dir = os.path.join(output_dir, "logs")

