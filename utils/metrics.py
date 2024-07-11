import numpy as np
import tensorflow as tf
import keras.backend as bk
from config import *
from utils.utility_functions import gram_schmit
from sklearn.metrics import r2_score
from scipy.linalg import norm
from utils.losses import InHouseLosses


# ********************* Custom Metrics *********************
class Custommetrics(object):

    def r2_metric(self, y_true, y_pred):
        if type(y_pred).__name__ == "EagerTensor":
            y_pred = y_pred.numpy()
        if type(y_true).__name__ == "EagerTensor":
            y_true = y_true.numpy()
        score = r2_score(y_true, y_pred)
        # score = r2_score(y_true.numpy(), y_pred.numpy())
        return score

    def geodesic_metric(self, y_true, y_pred):
        if type(y_pred).__name__ == "EagerTensor":
            y_pred = y_pred.numpy()
        if type(y_true).__name__ == "EagerTensor":
            y_true = y_true.numpy()
        vec1 = y_true.flatten()
        vec2 = y_pred.flatten()
        rot_mtrx1 = gram_schmit(vec1)
        rot_mtrx2 = gram_schmit(vec2)

        geodesic_error = np.degrees(np.arccos(np.sum(np.multiply(rot_mtrx1, rot_mtrx2)) /
                                              (np.linalg.norm(rot_mtrx1) * np.linalg.norm(rot_mtrx2))))
        return geodesic_error

    def is_invertible(self, a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
