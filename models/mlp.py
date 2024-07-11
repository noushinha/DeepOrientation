import os.path
import numpy as np
import pandas as pd
import shutil
import yaml
from utils.losses import InHouseLosses
from config import *
from utils.utility_functions import *
from utils.data_loader import *
from utils.metrics import Custommetrics
from utils.data_loader import *
from utils.plots import plot_euler_vectors
import time
from datetime import timedelta
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
# from utils import *
import keras
import tensorflow as tf
from keras.optimizers import Adam, SGD, RMSprop
from keras import models # Sequential, load_model
from keras.layers import Dense, Input, Flatten, LeakyReLU
from keras.metrics import MeanAbsoluteError, MeanSquaredError
from keras import Model, regularizers

tf.config.experimental_run_functions_eagerly(True)
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

class DeepOrt:
    def __init__(self):
        self.train_history = {'MSE': [], 'MAE': [], 'R2': [], 'GEO': []}
        self.valid_history = {'MSE': [], 'MAE': [], 'R2': [], 'GEO': []}

        self.train_steps = 0
        self.valid_steps = 0

        self.net = None
        self.tl_params = None

        self.job_id = 0
        self.checkpoint_file = ""
        self.res_dir = None
        self.dlobj = DataLoader()
        self.tr_time = None
        self.ts_time = None

    def start_model(self):
        self.prepare_data()
        self.get_model()
        if config_dict['transfer_learning']:
            self.load_pretrained_model()
            self.fit_model()
        elif config_dict['inference']:
            self.load_pretrained_model()
            self.inference("test")
        else:
            self.fit_model()

    def prepare_data(self):
        self.train_steps, self.valid_steps = self.dlobj.tr_steps, self.dlobj.vl_steps
        self.res_dir = os.path.join(log_dir, f"{self.job_id}")

        self.res_dir = os.path.join(log_dir, f"{self.job_id}")
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)
        shutil.copyfile(os.path.join(ROOT_DIR, "models/mlp.py"), os.path.join(self.res_dir, "mlp.txt"))
        shutil.copyfile(os.path.join(ROOT_DIR, "config.py"), os.path.join(self.res_dir, "config.txt"))

        print(f"SLURM Job ID = {self.job_id}")
        print(f"Steps per epoch during training ~ {self.train_steps}")
        print(f"Steps per epoch during validation ~ {self.valid_steps}")
        print(f"total number of samples in {config_dict['data_type']} = {self.dlobj.data.shape[0]}")
        print(f"Number of training samples in {config_dict['data_type']} ~ {self.train_steps * config_dict['batch_size_train']}")
        print(f"Number of validation samples in {config_dict['data_type']} ~ {self.valid_steps * config_dict['batch_size_valid']}")
        print(f"total number of test samples in {config_dict['data_type']} = {self.dlobj.ts_data.shape[0]}")

    def get_model(self):
        self.net = self.build_regression_network()

    def dense_block(self, num_nodes=128, l_name='fc1'):
        """
        returns a dense(fully connected) layer based on the given parameters
        num_nodes: number of nodes/units in this layer
        act_func: the activation function to be applied on this layer
        l_num: layer number to create a layer name
        inputs: data to be fed into this layer
        """
        return Dense(units=num_nodes,
                     kernel_initializer='random_normal',
                     kernel_regularizer=regularizers.L2(1e-4),
                     activation=LeakyReLU(alpha=0.1),
                     name=l_name)

    def build_regression_network(self):
        metrics_list = [MeanSquaredError(),
                        MeanAbsoluteError()]
        model = models.Sequential()
        model.add(Input(shape=(config_dict['patch_size'],
                               config_dict['patch_size'],
                               config_dict['patch_size'], 1), name='input_1'))

        model.add(Flatten())
        for i in range(len(config_dict['denselayer'])):
            model.add(self.dense_block(num_nodes=config_dict['denselayer'][i], l_name=f"fc{i + 1}"))
        model.add(Dense(units=config_dict['output_dim'],
                        activation="tanh",
                        name=f"fc{len(config_dict['denselayer'])+1}"))
        print(model.summary())

        s = InHouseLosses()
        optimiser = self.set_optimiser(config_dict['optname'])
        m = Custommetrics()

        metrics_list.append(m.r2_metric)
        if config_dict['representation_space'] == "6dof":
            metrics_list.append(m.geodesic_metric)

        # name of the loss function is evaluated from the parameter passed in config
        model.compile(loss=eval(config_dict['lossname']), optimizer=optimiser, metrics=metrics_list)

        return model

    def set_optimiser(self, opt_name):
        optimizer = Adam(learning_rate=config_dict['initial_lr'],
                         beta_1=.9, beta_2=.999, epsilon=1e-08, clipnorm=10)
        if opt_name == "SGD":
            optimizer = SGD(learning_rate=config_dict['initial_lr'], momentum=0.9, nesterov=True)
        elif opt_name == "RMSprop":
            optimizer = RMSprop(learning_rate=config_dict['initial_lr'], rho=0.9, epsilon=1e-08)

        return optimizer

    def fit_model(self):
        tr_results = {"geodesic_metric": [], "loss": [], "mean_absolute_error": [], "mean_squared_error": [],
                      "r2_metric": [], "val_geodesic_metric": [], "val_loss": [], "val_mean_absolute_error": [],
                      "val_mean_squared_error": [], "val_r2_metric": []}
        tr_start_time = time.time()
        print("Fit model called")
        self.checkpoint_file = f"weights_{config_dict['representation_space']}_{config_dict['num_epochs']}_" \
                               f"{config_dict['optname']}_{config_dict['initial_lr']}_{self.job_id}.weights.h5"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.res_dir, self.checkpoint_file),
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.res_dir,
                                                              histogram_freq=0,
                                                              write_graph=True,
                                                              write_images=False)
        earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

        x_train, x_valid, y_train, y_valid = self.dlobj.split_data()

        history = self.net.fit(x_train, y_train,
                               batch_size=config_dict['batch_size_train'],
                               epochs=config_dict['num_epochs'],
                               validation_data=(x_valid, y_valid),
                               validation_batch_size=config_dict['batch_size_valid'],
                               shuffle=True,
                               callbacks=[model_checkpoint_callback,
                                          tensorboard_callback,
                                          earlystopping_callback])

        for data_key, _ in history.history.items():
            df = pd.DataFrame(history.history[data_key])
            df.to_csv(os.path.join(self.res_dir, f"{data_key}_{self.job_id}.csv"), index=False)


        # tensorboard_callback
        self.net.save(os.path.join(self.res_dir, self.checkpoint_file))
        tr_end_time = time.time()
        self.time_calculation(tr_start_time, tr_end_time, "train")

        # predicting results on train data
        self.predict_res(self.dlobj.data, self.dlobj.labels, "train")

        # predicting results on test data
        self.predict_res(self.dlobj.ts_data, self.dlobj.ts_labels, "test")

        self.save_parameters()
        if config_dict['save_features'] or config_dict['save_filters']:
            self.extract_features("test")

    def time_calculation(self, start_time, end_time, mode):
        """
        start_time: the starting clock point
        end_time: the ending clock point
        mode: train or test
        """
        total_time = end_time - start_time
        deltatime = timedelta(seconds=total_time)
        print(f"{mode} execution time: {str(deltatime)}")
        if mode == "train":
            self.tr_time = str(deltatime)
        elif mode == "test":
            self.ts_time = str(deltatime)

    def predict_res(self, data, labels, mode):
        """
            numerical results and scores like mse, mae, ... are collected during training
            this function saves everything in csv format so later
            one can plot or study the behavior per epoch or per iteration
            for train and validation sets
        """
        if mode == "train":
            meta = self.dlobj.meta
            predict_batch_size = config_dict['batch_size_train']
        if mode == "test":
            meta = self.dlobj.ts_meta
            predict_batch_size = 1

        pred_labels = self.net.predict(data, batch_size=predict_batch_size)

        r2_scores = r2_score(labels.flatten(),
                             pred_labels.flatten(),
                             multioutput='variance_weighted')

        s = InHouseLosses()
        mse = s.mse(labels.flatten(), pred_labels.flatten()).numpy()

        estimated_orientations = []
        geodesic_err = []

        ts_start_time = time.time()

        for i in range(labels.shape[0]):
            gt = labels[i]
            pred = pred_labels[i]
            if config_dict['representation_space'] == "6dof":
                gt_6dof = gt
                pred_6dof = pred
                gt_rotmtrx = gram_schmit(gt_6dof)
                pred_rotmtrx = gram_schmit(pred_6dof)

                gt_euler = rotation_matrix_to_euler(gt_rotmtrx)
                pred_euler = rotation_matrix_to_euler(pred_rotmtrx)

                gt_quaternions = rotation_matrix_to_quaternion(gt_rotmtrx)
                pred_quaternions = rotation_matrix_to_quaternion(pred_rotmtrx)
            elif config_dict['representation_space'] == "euler":
                gt_euler = gt
                pred_euler = pred

                gt_rotmtrx = euler_to_rotation_matrix(gt_euler)
                pred_rotmtrx = euler_to_rotation_matrix(pred_euler)

                gt_quaternions = rotation_matrix_to_quaternion(gt_rotmtrx)
                pred_quaternions = rotation_matrix_to_quaternion(pred_rotmtrx)

                gt_6dof = np.hstack((gt_rotmtrx[:, 0], gt_rotmtrx[:, 1]))
                pred_6dof = np.hstack((pred_rotmtrx[:, 0], pred_rotmtrx[:, 1]))
            elif config_dict['representation_space'] == "quat":
                gt_quaternions = gt
                pred_quaternions = pred

                gt_rotmtrx = quaternion_to_rotation_matrix(gt_quaternions)
                pred_rotmtrx = quaternion_to_rotation_matrix(pred_quaternions)

                gt_euler = rotation_matrix_to_euler(gt_rotmtrx)
                pred_euler = rotation_matrix_to_euler(pred_rotmtrx)

                gt_6dof = np.hstack((gt_rotmtrx[:, 0], gt_rotmtrx[:, 1]))
                pred_6dof = np.hstack((pred_rotmtrx[:, 0], pred_rotmtrx[:, 1]))

            row = [gt_6dof, pred_6dof,
                   gt_rotmtrx.flatten(),
                   pred_rotmtrx.flatten(),
                   gt_euler, pred_euler,
                   gt_quaternions, pred_quaternions]
            estimated_orientations.append(row)

            if config_dict['representation_space'] == "6dof":
                geodesic_err.append(geodesic(gt_6dof, pred_6dof))

        result_dict = {"gt_6dof": [], "pred_6dof": [],
                       "gt_rotmtrx": [], "pred_rotmtrx": [],
                       "gt_euler": [], "pred_euler": [],
                       "gt_quaternions": [], "pred_quaternions": []}
        col_names = list(result_dict.keys())
        df = pd.DataFrame(estimated_orientations, columns=col_names)
        df.to_csv(os.path.join(self.res_dir, f"{mode}_estimated_orientations_{self.job_id}.csv"), index=False)

        if mode == "test":
            ts_end_time = time.time()
            self.time_calculation(ts_start_time, ts_end_time, "test")

        # prediction on a single sample
        rand_index = np.random.randint(0, len(estimated_orientations))
        for data_key, _ in result_dict.items():
            val = df.iloc[rand_index][data_key].tolist()
            result_dict[data_key].append(val)

        if config_dict['representation_space'] == "6dof":
            result_dict['geodesicerr'] = geodesic_err[rand_index].tolist()
        result_dict['r2'] = r2_scores  # .tolist()
        result_dict['mse'] = mse.tolist()

        for colname in col_names:
            header_name = colname.split("_")
            df_ort_subset = pd.DataFrame(df[colname])
            row = df_ort_subset.iloc[0][colname]
            col_list = [f"{header_name[1]}{i}" for i in range(len(row))]
            df_ort_subset[col_list[0:len(row)]] = pd.DataFrame(df_ort_subset[colname].tolist(),
                                                               index=df_ort_subset.index)
            df_ort_subset = df_ort_subset.drop(colname, axis=1)

            df_meta_subset = pd.DataFrame(meta, columns=['tomo_idx', 'x', 'y', 'z', 'class'])
            frames = [df_meta_subset, df_ort_subset]
            df_subset = pd.concat(frames, axis=1, join="inner")
            df_subset.to_csv(os.path.join(self.res_dir, f"{mode}_{colname}_{self.job_id}.csv"),
                             index=False)

            if colname == "pred_6dof":
                result_dict['mean_6dof'] = df_ort_subset.mean(axis=None).tolist()
                result_dict['std_6dof'] = df_ort_subset.values.std(ddof=1).tolist()

        with open(os.path.join(self.res_dir, f'{mode}_results_{self.job_id}.yml'), 'w') as outfile:
            yaml.dump(result_dict, outfile, default_flow_style=False)
        mol_vol = data[rand_index, :, :, :, 0]
        plot_euler_vectors(mol_vol,
                           np.reshape(df.iloc[rand_index]["gt_rotmtrx"], (3, 3)),
                           np.reshape(df.iloc[rand_index]["pred_rotmtrx"], (3, 3)),
                           self.job_id,
                           self.res_dir)

    def save_parameters(self):
        """
        this function dumps the configuration file used for training as a yaml file
        """
        del config_dict['denselayer']
        config_dict['jobid'] = self.job_id
        config_dict['train_exec_time'] = self.tr_time
        config_dict['test_exec_time'] = self.ts_time
        config_dict['train_data_size'] = self.dlobj.data.shape[0]
        config_dict['test_data_size'] = self.dlobj.ts_data.shape[0]
        with open(os.path.join(self.res_dir, f'params_{self.job_id}.yml'), 'w') as outfile:
            yaml.dump(config_dict, outfile, default_flow_style=False)

    def load_pretrained_model(self):
        """
        loads a pretrained model from the checkpoint directory set in config dictionary
        and turns off the trainable parameters of specific layers
        """
        print("Pretrained Model is loading...")
        if config_dict['checkpoint_dir'] != '':
            this_jobid = config_dict['checkpoint_dir'].split('/')[10].split('_')[0]
            with open(os.path.join(config_dict['checkpoint_dir'], f"params_{this_jobid}.yml")) as stream:
                self.tl_params = yaml.safe_load(stream)

            weghits_file_name = f"weights_{self.tl_params['representation_space']}_{self.tl_params['num_epochs']}" \
                                f"_{self.tl_params['optname']}_{self.tl_params['initial_lr']}_" \
                                f"{self.tl_params['jobid']}.keras"
            # weghits_file_name = "my_model.keras"
            m = Custommetrics()
            self.net = models.load_model(os.path.join(config_dict['checkpoint_dir'], weghits_file_name),
                                  custom_objects={"r2_metric":  m.r2_metric, "geodesic_metric": m.geodesic_metric})

            for layer in self.net.layers:
                if layer.name in config_dict['list_layers']:
                    layer.trainable = False

            self.net.summary()

    def extract_features(self, mode):
        """
        based on the config setting calls for two functions:
        one that extract learned features for specific layers (extract_layer_output)
        the other extracts learned filters for specific layers (extract_layer_filter
        """
        # you must save in the old jobid dir , retrieve the job id
        for layer in self.net.layers:
            if layer.name in config_dict['list_layers']:
                print(layer.name)
                if config_dict['save_features']:
                    pass_data = self.dlobj.data
                    if mode == "test":
                        pass_data = self.dlobj.ts_data
                    self.extract_layer_output(pass_data, mode, layer_name=layer.name)
                if config_dict['save_filters']:
                    self.extract_layer_filter(layer, layer_name=layer.name)

    def extract_layer_output(self, xinput, mode, layer_name="fc1"):
        """
        this function extract the features of a specific layer and saves them as npy file
        xinput: the data that should pass forward to the model so the features be extracted
        layer_name: name of the layer from which we need the extraction, default is first layer
        for a complete list of layers see the model summary printed in deeport_output_{job_id}.out files
        """
        intermediate_layer_model = Model(inputs=self.net.inputs, outputs=self.net.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(xinput)
        filename = str(mode) + "_" + str(layer_name) + "_Features"
        np.save(os.path.join(self.res_dir, filename), intermediate_output)

    def extract_layer_filter(self, ith_layer, layer_name="fc1"):
        """
        this function saves filters learned for a specific layer as npy file
        ith_layer: layer number (do not mix with layer name)
        layer_name: name of the layer from which we need the filters
        for a complete list of layers see the model summary printed in deeport_output_{job_id}.out files
        """
        filters, biases = ith_layer.get_weights()
        filename = str(layer_name) + "_Filters"
        np.save(os.path.join(self.res_dir, filename), filters)

    def inference(self, mode):
        estimated_orientations = []
        ts_start_time = time.time()

        meta = self.dlobj.meta
        data = self.dlobj.data
        predict_batch_size = config_dict['batch_size_train']
        if mode == "test":
            meta = self.dlobj.ts_meta
            predict_batch_size = 1
            data = self.dlobj.ts_data

        pred_labels = self.net.predict(data, batch_size=predict_batch_size)

        for i in range(pred_labels.shape[0]):
            pred_6dof = pred_labels[i]
            pred_rotmtrx = gram_schmit(pred_6dof)
            pred_euler = rotation_matrix_to_euler(pred_rotmtrx)
            pred_quaternions = rotation_matrix_to_quaternion(pred_rotmtrx)

            row = [pred_6dof, pred_rotmtrx.flatten(), pred_euler, pred_quaternions]
            estimated_orientations.append(row)

        result_dict = {"pred_6dof": [], "pred_rotmtrx": [], "pred_euler": [], "pred_quaternions": []}
        col_names = list(result_dict.keys())
        df = pd.DataFrame(estimated_orientations, columns=col_names)
        df.to_csv(os.path.join(self.res_dir, f"{mode}_estimated_orientations_{self.job_id}.csv"), index=False)

        if mode == "test":
            ts_end_time = time.time()
            self.time_calculation(ts_start_time, ts_end_time, "test")

        # prediction on a single sample
        rand_index = np.random.randint(0, len(estimated_orientations))
        for data_key, _ in result_dict.items():
            val = df.iloc[rand_index][data_key].tolist()
            result_dict[data_key].append(val)

        for colname in col_names:
            header_name = colname.split("_")
            df_ort_subset = pd.DataFrame(df[colname])
            row = df_ort_subset.iloc[0][colname]
            col_list = [f"{header_name[1]}{i}" for i in range(len(row))]
            df_ort_subset[col_list[0:len(row)]] = pd.DataFrame(df_ort_subset[colname].tolist(),
                                                               index=df_ort_subset.index)
            df_ort_subset = df_ort_subset.drop(colname, axis=1)

            df_meta_subset = pd.DataFrame(meta, columns=['tomo_idx', 'x', 'y', 'z', 'class'])
            frames = [df_meta_subset, df_ort_subset]
            df_subset = pd.concat(frames, axis=1, join="inner")
            df_subset.to_csv(os.path.join(self.res_dir, f"{mode}_{colname}_{self.job_id}.csv"),
                             index=False)

            if colname == "pred_6dof":
                result_dict['mean_6dof'] = df_ort_subset.mean(axis=None).tolist()
                result_dict['std_6dof'] = df_ort_subset.values.std(ddof=1).tolist()

        with open(os.path.join(self.res_dir, f'{mode}_results_{self.job_id}.yml'), 'w') as outfile:
            yaml.dump(result_dict, outfile, default_flow_style=False)

        if config_dict['save_features'] or config_dict['save_filters']:
            self.extract_features(mode)
