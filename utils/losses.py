import numpy as np
import tensorflow as tf
from keras.losses import MeanSquaredError, MeanAbsoluteError, binary_crossentropy, Huber # , Dice
import keras.backend as bk
from config import *
from utils.utility_functions import gram_schmit
from numpy import linalg

# ********************* Loss Functions *********************
class InHouseLosses(object):
    def __init__(self):
        print("\n")
        # print("semantic loss functions initialized")

    def mse(self, y_true, y_pred):
        tfmse = MeanSquaredError(reduction="sum")
        return tfmse(y_true, y_pred)

    def mae(self, y_true, y_pred):
        tfmae = MeanAbsoluteError(reduction="sum")
        return tfmae(y_true, y_pred)

    def huber(self, y_true, y_pred):
        tfhuber = Huber(delta=0.5)
        return tfhuber(y_true, y_pred)

    def l1_l2(self, y_true, y_pred):
        # L1 is MAE and L2 is MSE
        l1l2_loss = self.mae(y_true, y_pred) + self.mse(y_true, y_pred)
        return l1l2_loss / 2.0

    def gram_schmit(self, vec_6dof):
        # [N(a1)  N(a2-(b1.a2)b1) b1*b2]
        # N(a) = a / ||a||
        # a1: first column of 6DOF
        # a2: second column of 6DOF
        # b1, b2, b3 : columns of rotation matrix

        vec_6dof = tf.convert_to_tensor(vec_6dof, dtype=tf.float32)

        a1 = tf.convert_to_tensor(vec_6dof[:, 0:3], dtype=tf.float32)
        a2 = tf.convert_to_tensor(vec_6dof[:, 3:6], dtype=tf.float32)

        b1 = tf.math.divide(a1, tf.norm(a1))

        numerator2 = tf.math.subtract(a2, tf.math.multiply(tf.reduce_sum(tf.multiply(b1, a2)), b1))
        b2 = tf.math.divide(numerator2, tf.norm(numerator2))
        assert b1.shape[0] == b2.shape[0], "First and second column shapes does not match"
        batch_size = b2.shape[0]
        e1 = tf.convert_to_tensor(tf.repeat([[1., 0., 0.]], repeats=[batch_size], axis=0), dtype=tf.float32)
        e2 = tf.convert_to_tensor(tf.repeat([[0., 1., 0.]], repeats=[batch_size], axis=0), dtype=tf.float32)
        e3 = tf.convert_to_tensor(tf.repeat([[0., 0., 1.]], repeats=[batch_size], axis=0), dtype=tf.float32)

        b31 = tf.linalg.det(tf.transpose(tf.reshape(tf.concat([b1, b2, e1], axis=1), (batch_size, 3, 3)), perm=[0, 2, 1]))
        b32 = tf.linalg.det(tf.transpose(tf.reshape(tf.concat([b1, b2, e2], axis=1), (batch_size, 3, 3)), perm=[0, 2, 1]))
        b33 = tf.linalg.det(tf.transpose(tf.reshape(tf.concat([b1, b2, e3], axis=1), (batch_size, 3, 3)), perm=[0, 2, 1]))

        b3 = tf.convert_to_tensor(tf.transpose([b31, b32, b33]), dtype=tf.float32)

        rot_mtrx = tf.concat([b1, b2, b3], axis=1)

        return rot_mtrx

    def point_wise_l2(self, y_true, y_pred):

        rot_mtrx_true = self.gram_schmit(y_true)
        rot_mtrx_pred = self.gram_schmit(y_pred)

        vec_true = tf.reshape(rot_mtrx_true, (1, tf.math.reduce_prod(rot_mtrx_true.shape)))
        vec_pred = tf.reshape(rot_mtrx_pred, (1, tf.math.reduce_prod(rot_mtrx_pred.shape)))

        mse = tf.keras.losses.MeanSquaredError()
        point_wise_l2_loss = mse(vec_true, vec_pred)
        # point_wise_l2_loss = self.mse(y_true, y_pred)
        return point_wise_l2_loss
